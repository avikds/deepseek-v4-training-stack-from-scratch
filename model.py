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

# Step 18 - fp4_e2m1_quantize
def fp4_e2m1_quantize(x):
    # Representable non-negative FP4 E2M1 magnitudes.
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=x.dtype
    )

    # Work with magnitudes and clip values to the representable range.
    magnitude = torch.clamp(x.abs(), max=6.0)

    # Find the nearest representable magnitude.
    # torch.argmin returns the first minimum, so ties select
    # the smaller magnitude because levels are sorted ascending.
    distances = (magnitude.unsqueeze(-1) - levels).abs()
    nearest_indices = distances.argmin(dim=-1)
    quantized_magnitude = levels[nearest_indices]

    # Restore the original signs and preserve the input dtype.
    return torch.sign(x) * quantized_magnitude

# Step 19 - mxfp4_fake_quant
def mxfp4_fake_quant(w, block=32):
    # Perform all scale and quantization calculations on detached weights.
    w_detached = w.detach()
    original_shape = w.shape

    # Reshape into blocks of the specified size.
    blocks = w_detached.reshape(-1, block)

    # Find each block's maximum absolute value.
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    amax = amax.clamp(min=1e-12)

    # Compute the shared power-of-two scale:
    # scale = 2 ** (floor(log2(amax)) - 2)
    scale = torch.pow(
        2.0,
        torch.floor(torch.log2(amax)) - 2.0
    )

    # Quantize normalized weights to the FP4 E2M1 grid,
    # then rescale them to the original magnitude range.
    quantized = fp4_e2m1_quantize(blocks / scale) * scale

    # Restore the original tensor shape.
    quantized = quantized.reshape(original_shape)

    # Straight-through estimator: quantized forward values,
    # identity gradient with respect to the original weights.
    return w + (quantized - w).detach()

# Step 20 - fp8_fake_quant
def fp8_e4m3_quantize(x):
    # Work with magnitudes and clip to the maximum finite E4M3 value.
    magnitude = x.abs().clamp(max=448.0)

    # Avoid log2(0); zero values are restored correctly at the end.
    safe_magnitude = magnitude.clamp(min=1e-12)

    # Compute the exponent, with -6 as the minimum exponent.
    exponent = torch.floor(torch.log2(safe_magnitude)).clamp(min=-6.0)

    # E4M3 has three mantissa bits, so the quantization step is 2^(e-3).
    step = torch.pow(2.0, exponent - 3.0)

    # Round to the nearest representable multiple of the step.
    quantized_magnitude = torch.round(magnitude / step) * step

    # Restore the sign and preserve the input dtype and shape.
    return torch.sign(x) * quantized_magnitude


def fp8_fake_quant(w, block=128):
    # Perform scaling and quantization using detached weights.
    w_detached = w.detach()
    original_shape = w.shape

    # Reshape into blocks.
    blocks = w_detached.reshape(-1, block)

    # Compute one scale per block, avoiding division by zero.
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0

    # Quantize normalized weights and restore their original scale.
    quantized = fp8_e4m3_quantize(blocks / scale) * scale
    quantized = quantized.reshape(original_shape)

    # Straight-through estimator: quantized forward values,
    # identity gradient with respect to the original weights.
    return w + (quantized - w).detach()

# Step 21 - MTPModule
class MTPModule(nn.Module):
    def __init__(self, d_model, block):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_e = RMSNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = block

    def forward(self, h, tokens, embed):
        T = h.shape[1]

        h_prev = self.norm_h(h[:, :T - 2])
        e_next = self.norm_e(embed(tokens[:, 1:T - 1]))

        combined = torch.cat([h_prev, e_next], dim=-1)
        x = self.proj(combined)

        result = self.block(x)
        if isinstance(result, tuple):
            result = result[0]

        return result

# Step 22 - mtp_loss
def mtp_loss(h_mtp, head, tokens, weight=0.3):
    logits = head(h_mtp)
    targets = tokens[:, 2:]

    vocab = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1)
    )

    return weight * loss

# Step 23 - CausalAttention
class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.d_head)
        self.k_norm = RMSNorm(self.d_head)

    def forward(self, x):
        batch, T, d_model = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.reshape(
                batch, T, self.n_heads, self.d_head
            ).transpose(1, 2)

        q = self.q_norm(split_heads(q))
        k = self.k_norm(split_heads(k))
        v = split_heads(v)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        merged = attn_out.transpose(1, 2).contiguous().reshape(
            batch, T, d_model
        )
        return self.out(merged)

# Step 24 - TinyV4Decoder
class TinyV4Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        vocab = cfg["vocab"]
        d_model = cfg["d_model"]
        n_layers = cfg["n_layers"]
        n_heads = cfg["n_heads"]
        d_ff = cfg["d_ff"]
        n_experts = cfg["n_experts"]
        k = cfg["k"]
        n_hc = cfg["n_hc"]
        hash_layers = cfg["hash_layers"]
        sinkhorn_iters = cfg["sinkhorn_iters"]
        constrained = cfg["constrained"]
        clamp = cfg["clamp"]

        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList()

        for i in range(n_layers):
            attention = CausalAttention(d_model, n_heads)
            self.blocks.append(
                MHCBlock(
                    d_model, n_hc, attention,
                    constrained=constrained,
                    sinkhorn_iters=sinkhorn_iters
                )
            )

            moe = MoELayer(
                d_model, d_ff, n_experts, k,
                hash_routing=(i < hash_layers),
                clamp=clamp
            )
            self.blocks.append(
                MHCBlock(
                    d_model, n_hc, moe,
                    constrained=constrained,
                    sinkhorn_iters=sinkhorn_iters
                )
            )

        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

        if cfg["mtp"]:
            self.mtp = MTPModule(
                d_model,
                nn.Sequential(
                    RMSNorm(d_model),
                    nn.Linear(d_model, d_model, bias=False)
                )
            )
        else:
            self.mtp = None

    def forward(self, tokens, routing_plan=None):
        x = expand_residual(self.embed(tokens), self.cfg["n_hc"])
        infos = []
        moe_index = 0

        for block_index, block in enumerate(self.blocks):
            is_moe_block = (block_index % 2 == 1)

            if is_moe_block:
                kwargs = {"token_ids": tokens}

                if routing_plan is not None:
                    kwargs["indices"] = routing_plan[moe_index]

                x, info = block(x, **kwargs)
                infos.append(info)
                moe_index += 1
            else:
                x = block(x)

        h = self.final_norm(collapse_residual(x))
        return h, infos

# Step 25 - synthetic_tokens
def synthetic_tokens(vocab, n_seq, seq_len, seed, branching=4):
    g = torch.Generator().manual_seed(seed)

    succ = torch.randint(
        0, vocab, (vocab, branching), generator=g
    )

    tokens = torch.empty((n_seq, seq_len), dtype=torch.long)
    if seq_len == 0:
        return tokens

    tokens[:, 0] = torch.randint(
        0, vocab, (n_seq,), generator=g
    )

    for t in range(1, seq_len):
        choice = torch.randint(
            0, branching, (n_seq,), generator=g
        )
        tokens[:, t] = succ[tokens[:, t - 1], choice]

    return tokens


def count_loss_spikes(losses, window=20, factor=1.5):
    if window <= 0:
        raise ValueError("window must be positive")

    spike_count = 0

    for i in range(window, len(losses)):
        previous = sorted(losses[i - window:i])
        median = previous[window // 2]

        if losses[i] > factor * median:
            spike_count += 1

    return int(spike_count)

# Step 26 - train_tiny_v4
import copy

def fake_quant_params(model):
    # Return a parameter-name -> tensor mapping for functional_call.
    quantized_params = {}

    for name, param in model.named_parameters():
        # Routed expert weights use MXFP4 fake quantization.
        if any(key in name for key in ("w_gate", "w_up", "w_down")):
            quantized_params[name] = mxfp4_fake_quant(param)

        # Other eligible 2-D weights use FP8 E4M3 fake quantization.
        elif (
            param.ndim == 2
            and param.numel() % 128 == 0
            and not any(
                key in name
                for key in ("embed", "head", "centroids", "static")
            )
        ):
            quantized_params[name] = fp8_fake_quant(param)

        # Leave all remaining parameters unchanged.
        else:
            quantized_params[name] = param

    return quantized_params


def train_tiny_v4(
    cfg,
    data,
    n_steps,
    lr,
    seed,
    batch=16,
    delay=0,
    bias_balance=True,
    bias_speed=0.001,
    qat=False,
    model=None
):
    # Seed model initialization and other global PyTorch randomness.
    torch.manual_seed(seed)

    # Construct the model unless an existing model was supplied.
    if model is None:
        model = TinyV4Decoder(cfg)

    model.train()

    # Keep a historical model for anticipatory routing when requested.
    hist = copy.deepcopy(model) if delay > 0 else None

    # Split trainable parameters into Muon and AdamW groups.
    muon, adamw = split_parameters(model)

    # Optimizer state and an independently seeded batch sampler.
    state = {}
    g = torch.Generator().manual_seed(seed)

    losses = []
    muon_rms = []
    adamw_rms = []
    agreement = []
    loads = []

    # MoE layers are stored at odd positions in the decoder's block list.
    moe_layers = [
        block.layer
        for i, block in enumerate(model.blocks)
        if i % 2 == 1
    ]

    for step in range(n_steps):
        # Sample a batch of sequences, allowing repeated rows.
        row_indices = torch.randint(
            0, len(data), (batch,), generator=g
        )
        tokens = data[row_indices.to(data.device)]

        # Refresh the historical model at the requested interval.
        if delay > 0:
            if step % delay == 0:
                hist.load_state_dict(model.state_dict())

            # Compute expert assignments using historical parameters.
            plan = anticipatory_routing_plan(hist, tokens)
        else:
            plan = None

        # Forward pass, optionally using fake-quantized parameters.
        if qat:
            h, infos = torch.func.functional_call(
                model,
                fake_quant_params(model),
                (tokens,),
                {"routing_plan": plan}
            )
        else:
            h, infos = model(tokens, routing_plan=plan)

        # Main next-token prediction loss.
        logits = model.head(h[:, :-1, :])
        vocab = logits.shape[-1]

        loss = F.cross_entropy(
            logits.reshape(-1, vocab),
            tokens[:, 1:].reshape(-1)
        )

        # Include the multi-token prediction auxiliary objective.
        total = loss
        if model.mtp is not None:
            h_mtp = model.mtp(h, tokens, model.embed)
            total = total + mtp_loss(
                h_mtp, model.head, tokens, weight=0.3
            )

        # Add the slight sequence-wise balance loss for each MoE layer.
        for info in infos:
            total = total + 1e-4 * sequence_balance_loss(
                info["scores"],
                info["indices"],
                cfg["k"]
            )

        # Backpropagate the combined objective.
        model.zero_grad()
        total.backward()

        # Apply one combined Muon and AdamW optimizer step.
        stats = optimizer_step(
            muon,
            adamw,
            state,
            lr
        )

        # Record the primary next-token loss and update diagnostics.
        losses.append(float(loss.detach()))
        muon_rms.append(stats["muon_rms"])
        adamw_rms.append(stats["adamw_rms"])

        # Record this step's per-layer expert loads.
        loads = [
            info["load"].tolist()
            for info in infos
        ]

        # Compare historical routing with current learned routing.
        if delay > 0:
            historical_learned = []
            current_learned = []

            for j, (layer, info) in enumerate(zip(moe_layers, infos)):
                # Hash-routed layers do not use learned top-k routing.
                if layer.hash_routing:
                    continue

                historical_learned.append(plan[j])

                # These are the choices current parameters would make,
                # using scores from the same forward pass.
                current_indices, _ = route_tokens(
                    info["scores"],
                    layer.bias,
                    cfg["k"]
                )
                current_learned.append(current_indices)

            agreement.append(
                routing_agreement(
                    historical_learned,
                    current_learned
                )
            )

        # Update auxiliary-loss-free routing biases after the step.
        with torch.no_grad():
            if bias_balance:
                for layer, info in zip(moe_layers, infos):
                    if not layer.hash_routing:
                        layer.bias.copy_(
                            update_routing_bias(
                                layer.bias,
                                info["indices"],
                                cfg["n_experts"],
                                bias_speed
                            )
                        )

    return {
        "model": model,
        "losses": losses,
        "spikes": count_loss_spikes(losses),
        "muon_rms": muon_rms,
        "adamw_rms": adamw_rms,
        "loads": loads,
        "agreement": agreement
    }

