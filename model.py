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

