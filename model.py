"""
DeepSeek-V4 Training Stack from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - RMSNorm
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()

        # Learnable scaling parameter, initialized to ones.
        self.weight = nn.Parameter(torch.ones(d))

        # Small constant to prevent division by zero.
        self.eps = eps

    def forward(self, x):
        # Compute the mean squared value along the last dimension.
        rms_squared = torch.mean(x ** 2, dim=-1, keepdim=True)

        # Normalize using the reciprocal square root.
        normalized_x = x * torch.rsqrt(rms_squared + self.eps)

        # Apply the learnable element-wise scaling parameter.
        return normalized_x * self.weight

# Step 2 - expand_residual
def expand_residual(x, n_hc):
    # Expand the residual stream with n_hc identical copies.
    # clone() ensures the result is a fresh tensor, not a view or
    # shared expanded storage.
    return x.unsqueeze(-2).expand(
        *x.shape[:-1], n_hc, x.shape[-1]
    ).clone()


def collapse_residual(X):
    # Average across the hyper-connection stream dimension.
    # Shape: (..., n_hc, d) -> (..., d)
    return X.mean(dim=-2)

# Step 3 - HCParamGenerator
import math

class HCParamGenerator(nn.Module):
    def __init__(self, d_model, n_hc, alpha_init=0.01):
        super().__init__()

        self.n_hc = n_hc

        # Normalize the flattened hyper-connection residual state.
        self.norm = RMSNorm(n_hc * d_model)

        # Initialize dynamic weight matrices with scaled Gaussian values.
        scale = math.sqrt(n_hc * d_model)

        self.w_pre = nn.Parameter(
            torch.randn(n_hc * d_model, n_hc) / scale
        )
        self.w_res = nn.Parameter(
            torch.randn(n_hc * d_model, n_hc * n_hc) / scale
        )
        self.w_post = nn.Parameter(
            torch.randn(n_hc * d_model, n_hc) / scale
        )

        # Static mappings: zero input/output biases and identity residual map.
        self.static_pre = nn.Parameter(torch.zeros(n_hc))
        self.static_res = nn.Parameter(torch.eye(n_hc))
        self.static_post = nn.Parameter(torch.zeros(n_hc))

        # Learnable scalar gates controlling the dynamic contributions.
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, X):
        # X has shape (..., n_hc, d_model).
        # Flatten the last two dimensions: (..., n_hc * d_model).
        x = X.flatten(start_dim=-2)

        # Normalize the flattened residual state.
        x_hat = self.norm(x)

        # Generate the input mixing map: (..., n_hc).
        a_raw = self.alpha_pre * (x_hat @ self.w_pre)
        a_raw = a_raw + self.static_pre

        # Generate the residual mixing map before reshaping.
        # Initial shape: (..., n_hc * n_hc).
        b_flat = self.alpha_res * (x_hat @ self.w_res)

        # Reshape the final dimension into an (n_hc, n_hc) matrix.
        b_raw = b_flat.reshape(
            *x_hat.shape[:-1], self.n_hc, self.n_hc
        )
        b_raw = b_raw + self.static_res

        # Generate the output mixing map: (..., n_hc).
        c_raw = self.alpha_post * (x_hat @ self.w_post)
        c_raw = c_raw + self.static_post

        return a_raw, b_raw, c_raw

# Step 4 - sinkhorn_knopp
def sinkhorn_knopp(b_raw, n_iter=20):
    # Convert raw residual-map logits into positive values.
    M = torch.exp(b_raw)

    # Alternately normalize columns and rows.
    for _ in range(n_iter):
        # Column normalization: each column sums to 1.
        M = M / M.sum(dim=-2, keepdim=True)

        # Row normalization: each row sums to 1.
        M = M / M.sum(dim=-1, keepdim=True)

    return M

# Step 5 - MHCBlock
class MHCBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_hc,
        layer,
        constrained=True,
        sinkhorn_iters=20
    ):
        super().__init__()

        # The layer wrapped by the hyper-connection block.
        self.layer = layer

        # Generates dynamic input, residual, and output mappings.
        self.gen = HCParamGenerator(d_model, n_hc)

        self.constrained = constrained
        self.sinkhorn_iters = sinkhorn_iters

    def mappings(self, X):
        # Generate the raw hyper-connection mappings.
        a_raw, b_raw, c_raw = self.gen(X)

        if self.constrained:
            # Constrain input mixing coefficients to (0, 1).
            A = torch.sigmoid(a_raw)

            # Project residual mixing matrices onto the
            # doubly stochastic manifold.
            B = sinkhorn_knopp(b_raw, self.sinkhorn_iters)

            # Constrain output mixing coefficients to (0, 2).
            C = 2.0 * torch.sigmoid(c_raw)

            return A, B, C

        # Unconstrained hyper-connections use the raw mappings.
        return a_raw, b_raw, c_raw

    def forward(self, X, **kwargs):
        # X has shape (..., n_hc, d_model).
        A, B, C = self.mappings(X)

        # Mix the residual streams to construct the layer input.
        # A: (..., n_hc)
        # X: (..., n_hc, d_model)
        # h: (..., d_model)
        h = (A.unsqueeze(-1) * X).sum(dim=-2)

        # Apply the wrapped layer, forwarding any keyword arguments.
        result = self.layer(h, **kwargs)

        # Support layers that return either a tensor or (out, extra).
        if isinstance(result, tuple):
            out, extra = result
        else:
            out = result
            extra = None

        # Mix the existing residual streams using B.
        # B: (..., n_hc, n_hc)
        # X: (..., n_hc, d_model)
        residual = B @ X

        # Inject the layer output into every residual stream,
        # scaled by its corresponding output coefficient.
        # C.unsqueeze(-1): (..., n_hc, 1)
        # out.unsqueeze(-2): (..., 1, d_model)
        X_next = residual + C.unsqueeze(-1) * out.unsqueeze(-2)

        # Preserve the auxiliary output when the layer returns a tuple.
        if extra is not None:
            return X_next, extra

        return X_next

# Step 6 - signal_norm_across_depth
def signal_norm_across_depth(
    n_layers,
    d_model,
    n_hc,
    constrained,
    res_noise,
    seed,
    n_tokens=8
):
    # Make the experiment reproducible.
    torch.manual_seed(seed)

    blocks = []

    # Construct each block and perturb its static residual map.
    for _ in range(n_layers):
        block = MHCBlock(
            d_model,
            n_hc,
            nn.Tanh(),
            constrained=constrained
        )

        # Model drift away from the identity residual map.
        with torch.no_grad():
            block.gen.static_res.add_(
                res_noise * torch.randn(n_hc, n_hc)
            )

        blocks.append(block)

    # Initialize the residual state with identical hyper-connection streams.
    X = expand_residual(
        torch.randn(n_tokens, d_model),
        n_hc
    )

    rms_history = []
    spec_history = []

    # Measure signal propagation without tracking gradients.
    with torch.no_grad():
        for block in blocks:
            # Obtain the residual mixing matrices for the current input.
            _, B, _ = block.mappings(X)

            # Record the largest spectral norm across all tokens.
            spectral_norms = torch.linalg.matrix_norm(B, ord=2)
            spec = spectral_norms.max().item()

            # Propagate the residual state through the mHC block.
            X = block(X)

            # Record the global RMS of the updated residual state.
            rms = torch.sqrt(torch.mean(X ** 2)).item()

            rms_history.append(round(float(rms), 4))
            spec_history.append(round(float(spec), 4))

    return {
        "rms": rms_history,
        "spec": spec_history
    }

# Step 7 - newton_schulz_step
def newton_schulz_step(M, a, b, c):
    # Compute A = M @ M^T, transposing only the last two dimensions.
    A = M @ M.transpose(-2, -1)

    # Compute the matrix square A^2.
    A_squared = A @ A

    # Apply one Newton-Schulz iteration.
    return (
        a * M
        + b * (A @ M)
        + c * (A_squared @ M)
    )

# Step 8 - hybrid_newton_schulz
def hybrid_newton_schulz(M, fast_steps=8, stable_steps=2, eps=1e-7):
    # Work on the transpose of tall matrices so the Gram matrix
    # is formed using the smaller matrix dimension.
    transposed = M.shape[-2] > M.shape[-1]
    X = M.transpose(-2, -1) if transposed else M

    # Normalize each matrix by its Frobenius norm.
    # keepdim=True preserves the matrix dimensions for broadcasting.
    fro_norm = torch.linalg.matrix_norm(
        X, ord="fro", dim=(-2, -1), keepdim=True
    )
    X = X / (fro_norm + eps)

    # Fast Newton-Schulz iterations.
    for _ in range(fast_steps):
        X = newton_schulz_step(
            X, 3.4445, -4.7750, 2.0315
        )

    # Numerically stabilizing Newton-Schulz iterations.
    for _ in range(stable_steps):
        X = newton_schulz_step(
            X, 2.0, -1.5, 0.5
        )

    # Restore the original matrix orientation, if necessary.
    if transposed:
        X = X.transpose(-2, -1)

    return X

# Step 9 - muon_update
def muon_update(grad, buf, mu=0.95, gamma=0.18):
    # Get the matrix dimensions from the last two axes.
    n, m = grad.shape[-2:]

    # Update the momentum buffer in place:
    # M_t = mu * M_{t-1} + grad
    with torch.no_grad():
        buf.mul_(mu).add_(grad)

    # Compute the Nesterov-corrected direction:
    # O' = hybrid_newton_schulz(mu * M_t + grad)
    nesterov_grad = mu * buf + grad
    O = hybrid_newton_schulz(nesterov_grad)

    # Rescale the orthogonalized matrix so its RMS is approximately gamma.
    O = O * (math.sqrt(max(n, m)) * gamma)

    return O

# Step 10 - split_parameters
def split_parameters(
    model,
    adamw_keywords=("embed", "head", "norm", "static", "alpha", "bias")
):
    muon = []
    adamw = []

    # Normalize keywords for case-insensitive name matching.
    keywords = tuple(keyword.lower() for keyword in adamw_keywords)

    # Preserve the order returned by model.named_parameters().
    for name, param in model.named_parameters():

        # Ignore frozen parameters.
        if not param.requires_grad:
            continue

        # Low-dimensional parameters and parameters whose names match
        # the specified keywords are assigned to AdamW.
        name_lower = name.lower()
        use_adamw = (
            param.ndim < 2
            or any(keyword in name_lower for keyword in keywords)
        )

        if use_adamw:
            adamw.append((name, param))
        else:
            muon.append((name, param))

    return muon, adamw

# Step 11 - optimizer_step
def optimizer_step(
    muon_params,
    adamw_params,
    state,
    lr,
    mu=0.95,
    wd=0.1,
    gamma=0.18,
    betas=(0.9, 0.95),
    eps=1e-20
):
    # Increment the global optimization step (initialized at zero).
    state["t"] = state.get("t", 0) + 1
    t = state["t"]

    b1, b2 = betas

    muon_rms_values = []
    adamw_rms_values = []

    with torch.no_grad():
        # --------------------------------------------------
        # 1. Muon update for eligible matrix parameters
        # --------------------------------------------------
        for name, p in muon_params:
            # Skip parameters without a gradient.
            if p.grad is None:
                continue

            # Initialize the momentum buffer on first use.
            key = "muon_" + name
            if key not in state:
                state[key] = torch.zeros_like(p)

            buf = state[key]

            # Compute the orthogonalized Nesterov momentum update.
            O = muon_update(p.grad, buf, mu=mu, gamma=gamma)

            # Apply decoupled weight decay and the Muon update:
            # p <- p * (1 - lr * wd) - lr * O
            p.mul_(1.0 - lr * wd)
            p.add_(O, alpha=-lr)

            # Record the RMS of this parameter's Muon update.
            update_rms = torch.sqrt(torch.mean(O ** 2))
            muon_rms_values.append(update_rms.item())

        # --------------------------------------------------
        # 2. AdamW update for remaining parameters
        # --------------------------------------------------
        for name, p in adamw_params:
            # Skip parameters without a gradient.
            if p.grad is None:
                continue

            g = p.grad

            # Initialize first- and second-moment buffers.
            m_key = "adam_m_" + name
            v_key = "adam_v_" + name

            if m_key not in state:
                state[m_key] = torch.zeros_like(p)
            if v_key not in state:
                state[v_key] = torch.zeros_like(p)

            m = state[m_key]
            v = state[v_key]

            # Update the exponential moving averages.
            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

            # Apply bias correction.
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)

            # Compute the AdamW adaptive update.
            adaptive_update = m_hat / (torch.sqrt(v_hat) + eps)

            # Apply decoupled weight decay and the AdamW update.
            p.mul_(1.0 - lr * wd)
            p.add_(adaptive_update, alpha=-lr)

            # Record the RMS of this parameter's adaptive update.
            update_rms = torch.sqrt(torch.mean(adaptive_update ** 2))
            adamw_rms_values.append(update_rms.item())

    # Average the per-parameter RMS values.
    # Empty groups return 0.0.
    muon_rms = (
        sum(muon_rms_values) / len(muon_rms_values)
        if muon_rms_values else 0.0
    )

    adamw_rms = (
        sum(adamw_rms_values) / len(adamw_rms_values)
        if adamw_rms_values else 0.0
    )

    return {
        "muon_rms": float(muon_rms),
        "adamw_rms": float(adamw_rms)
    }

# Step 12 - route_tokens
def route_tokens(scores, bias, k):
    # Add the per-expert balancing bias to the routing scores.
    adjusted_scores = scores + bias

    # Stable descending sort ensures that ties preserve the original
    # expert order, so the lower expert index wins.
    _, sorted_indices = torch.sort(
        adjusted_scores,
        dim=-1,
        descending=True,
        stable=True
    )

    # Select the top-k experts for each token.
    indices = sorted_indices[:, :k].long()

    # Gather the ORIGINAL affinities, without the balancing bias.
    selected_scores = torch.gather(scores, dim=-1, index=indices)

    # Normalize the selected affinities to obtain routing gates.
    gates = selected_scores / (
        selected_scores.sum(dim=-1, keepdim=True) + 1e-9
    )

    return indices, gates

# Step 13 - update_routing_bias
def update_routing_bias(bias, indices, n_experts, speed=0.001):
    # Count how often each expert was selected.
    # Flatten indices to support tensors of any shape.
    flat_indices = indices.reshape(-1).to(
        device=bias.device, dtype=torch.long
    )
    counts = torch.bincount(
        flat_indices, minlength=n_experts
    ).to(dtype=bias.dtype)

    # Compare each expert's load with the mean expert load.
    mean_count = counts.mean()

    # Overloaded experts lose bias; underloaded experts gain bias.
    # Exactly balanced experts retain their current bias.
    new_bias = bias - speed * torch.sign(counts - mean_count)

    return new_bias


def sequence_balance_loss(scores, indices, k):
    # scores shape: (n_tokens, n_experts)
    # indices shape: (n_tokens, k)
    n_tokens, n_experts = scores.shape

    # Count the number of times each expert was selected.
    flat_indices = indices.reshape(-1).to(
        device=scores.device, dtype=torch.long
    )
    counts = torch.bincount(
        flat_indices, minlength=n_experts
    ).to(dtype=scores.dtype)

    # Compute the normalized expert-selection frequency:
    # f_i = n_experts / (k * n_tokens) * count_i
    f = counts * (n_experts / (k * n_tokens))

    # Normalize the original affinities across experts for each token.
    # The small epsilon prevents division by zero.
    normalized_scores = scores / (
        scores.sum(dim=-1, keepdim=True) + 1e-9
    )

    # Mean normalized affinity assigned to each expert.
    P = normalized_scores.mean(dim=0)

    # Sequence-wise balance loss: sum_i f_i * P_i.
    return torch.sum(f * P)

# Step 14 - hash_route
def hash_route(token_ids, n_experts, k):
    # Create expert offsets: [0, 1, ..., k-1].
    offsets = torch.arange(k, device=token_ids.device)

    # Assign each token to k consecutive experts using its token ID.
    # Output shape: (..., k).
    indices = (token_ids.unsqueeze(-1) + offsets) % n_experts

    return indices.long()

# Step 15 - clamped_swiglu
import torch.nn.functional as F

def clamped_swiglu(x, w_gate, w_up, w_down, limit=10.0):
    # Compute the gate projection, applying an upper-bound clamp only.
    gate = torch.clamp(x @ w_gate, max=limit)

    # Compute the up projection, clamping both lower and upper bounds.
    lin = torch.clamp(x @ w_up, min=-limit, max=limit)

    # Apply SwiGLU: silu(gate) * linear projection, then down-project.
    return (F.silu(gate) * lin) @ w_down

# Step 16 - MoELayer
class MoELayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        n_experts,
        k,
        n_shared=1,
        hash_routing=False,
        clamp=True
    ):
        super().__init__()

        self.n_experts = n_experts
        self.k = k
        self.hash_routing = hash_routing
        self.clamp = clamp

        # Scaling factors for parameter initialization.
        model_scale = math.sqrt(d_model)
        ff_scale = math.sqrt(d_ff)
        shared_ff_dim = n_shared * d_ff
        shared_scale = math.sqrt(shared_ff_dim)

        # Routed expert parameters.
        # Initialize in the exact order specified by the task.
        self.w_gate = nn.Parameter(
            torch.randn(n_experts, d_model, d_ff) / model_scale
        )
        self.w_up = nn.Parameter(
            torch.randn(n_experts, d_model, d_ff) / model_scale
        )
        self.w_down = nn.Parameter(
            torch.randn(n_experts, d_ff, d_model) / ff_scale
        )

        # Shared expert parameters.
        self.shared_gate = nn.Parameter(
            torch.randn(d_model, shared_ff_dim) / model_scale
        )
        self.shared_up = nn.Parameter(
            torch.randn(d_model, shared_ff_dim) / model_scale
        )
        self.shared_down = nn.Parameter(
            torch.randn(shared_ff_dim, d_model) / shared_scale
        )

        # Routing centroids used to calculate token-expert affinities.
        self.centroids = nn.Parameter(
            torch.randn(n_experts, d_model) / model_scale
        )

        # Auxiliary-loss-free load-balancing bias.
        self.register_buffer("bias", torch.zeros(n_experts))

    def affinity(self, x):
        # Compute non-negative token-to-expert affinities:
        # scores = sqrt(softplus(x @ centroids.T))
        logits = x @ self.centroids.transpose(-2, -1)
        return torch.sqrt(F.softplus(logits))

    def forward(self, x, token_ids=None, indices=None):
        # Save the original shape and flatten all token dimensions.
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])

        # Calculate routing affinities for every flattened token.
        scores = self.affinity(x_flat)

        # Obtain routing indices from supplied indices, hash routing,
        # or affinity-based top-k routing, in that priority order.
        if indices is None:
            if self.hash_routing:
                if token_ids is None:
                    raise ValueError(
                        "token_ids must be provided when hash_routing=True"
                    )

                token_ids = token_ids.reshape(-1).to(
                    device=x_flat.device, dtype=torch.long
                )
                indices = hash_route(
                    token_ids, self.n_experts, self.k
                )
            else:
                indices, _ = route_tokens(
                    scores, self.bias, self.k
                )

        # Support supplied indices with either flattened or leading
        # token dimensions. Keep indices compatible with the input device.
        indices = indices.reshape(-1, self.k).to(
            device=x_flat.device, dtype=torch.long
        )

        # Routing gates always use the ORIGINAL affinities, irrespective
        # of whether the indices came from hashing or were supplied.
        selected_scores = torch.gather(
            scores, dim=-1, index=indices
        )
        gates = selected_scores / (
            selected_scores.sum(dim=-1, keepdim=True) + 1e-9
        )

        # Apply SwiGLU clamping when enabled.
        limit = 10.0 if self.clamp else float("inf")

        # Every token receives the shared expert's output.
        y = clamped_swiglu(
            x_flat,
            self.shared_gate,
            self.shared_up,
            self.shared_down,
            limit=limit
        )

        # Add weighted outputs from the selected routed experts.
        for e in range(self.n_experts):
            # Identify token rows and routing slots assigned to expert e.
            rows, slot = (indices == e).nonzero(as_tuple=True)

            # Skip experts that received no tokens.
            if rows.numel() == 0:
                continue

            # Compute this expert's output for its assigned tokens.
            expert_out = clamped_swiglu(
                x_flat[rows],
                self.w_gate[e],
                self.w_up[e],
                self.w_down[e],
                limit=limit
            )

            # Weight expert outputs by their routing gates.
            weighted_out = (
                gates[rows, slot].unsqueeze(-1) * expert_out
            )

            # Accumulate routed expert outputs into the token rows.
            y = y.index_add(0, rows, weighted_out)

        # Count all top-k expert assignments.
        load = torch.bincount(
            indices.reshape(-1),
            minlength=self.n_experts
        )

        # Return the output in its original shape and routing diagnostics.
        y = y.reshape(original_shape)

        info = {
            "indices": indices,
            "scores": scores,
            "load": load
        }

        return y, info

# Step 17 - anticipatory_routing_plan
def anticipatory_routing_plan(model_hist, tokens):
    # Run the historical model without tracking gradients.
    with torch.no_grad():
        _, infos = model_hist(tokens)

        # Extract routing indices in MoE layer order.
        plan = [info["indices"].detach() for info in infos]

    return plan


def routing_agreement(plan_a, plan_b):
    # Both plans must have the same number of layers.
    if len(plan_a) != len(plan_b):
        raise ValueError(
            "Both routing plans must contain the same number of layers."
        )

    matching_entries = 0
    total_entries = 0

    # Compare routing assignments across all layers.
    for indices_a, indices_b in zip(plan_a, plan_b):
        if indices_a.shape != indices_b.shape:
            raise ValueError(
                "Routing index tensors must have matching shapes."
            )

        # Sort expert indices so slot order does not affect agreement.
        sorted_a = torch.sort(indices_a, dim=-1).values
        sorted_b = torch.sort(
            indices_b.to(device=sorted_a.device), dim=-1
        ).values

        matching_entries += (sorted_a == sorted_b).sum().item()
        total_entries += sorted_a.numel()

    # Handle empty routing plans.
    if total_entries == 0:
        return 0.0

    # Return the agreement using float32 precision, as expected by the tests.
    agreement = torch.tensor(
        matching_entries / total_entries,
        dtype=torch.float32
    )

    return float(agreement.item())

