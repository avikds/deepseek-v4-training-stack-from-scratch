"""
DeepSeek-V4 Training Stack from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""DeepSeek-V4 training stack at toy scale: mHC signal propagation, hybrid Newton-Schulz, Muon update RMS,
routing balance and SwiGLU clamping, MXFP4 versus FP8 error, and two short seeded training runs of a tiny
V4-style decoder (stability mechanisms off and on) followed by an FP4 quantization-aware fine-tuning phase."""
import time
import torch
import torch.nn.functional as F


def eval_loss(model, tokens, quantized=False):
    with torch.no_grad():
        if quantized:
            h, _ = torch.func.functional_call(model, fake_quant_params(model), (tokens,), {"routing_plan": None})
        else:
            h, _ = model(tokens)
        logits = model.head(h[:, :-1])
        return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)))


def clamp_diagnostics(model, tokens):
    """Largest |linear pre-activation| the SwiGLU clamp sees in each MoE layer and the fraction it clips."""
    captured = []
    hooks = [blk.layer.register_forward_pre_hook(lambda m, args, kw, c=captured: c.append(args[0].reshape(-1, args[0].shape[-1])), with_kwargs=True)
             for blk in model.blocks if isinstance(blk.layer, MoELayer)]
    with torch.no_grad():
        model(tokens)
    for hk in hooks:
        hk.remove()
    out = []
    for blk, x in zip([b for b in model.blocks if isinstance(b.layer, MoELayer)], captured):
        layer = blk.layer
        with torch.no_grad():
            pre = torch.cat([x @ layer.shared_up] + [x @ layer.w_up[e] for e in range(layer.n_experts)], dim=-1)
        out.append((round(float(pre.abs().max()), 2), round(float((pre.abs() > 10).float().mean()), 4)))
    return out


def spectral_diagnostics(model, tokens):
    """Largest spectral norm of the residual mixing map B over all tokens and blocks of a trained model."""
    worst = 0.0
    with torch.no_grad():
        X = expand_residual(model.embed(tokens), model.cfg["n_hc"])
        for blk in model.blocks:
            B = blk.mappings(X)[1]
            worst = max(worst, float(torch.linalg.matrix_norm(B, ord=2).max()))
            out = blk(X, token_ids=tokens) if isinstance(blk.layer, MoELayer) else blk(X)
            X = out[0] if isinstance(out, tuple) else out
    return round(worst, 4)


def main():
    t0 = time.time()
    torch.manual_seed(0)

    print("=== Part 1: signal norm across 24 blocks, static residual maps perturbed by N(0, 0.5) ===")
    for constrained in (False, True):
        r = signal_norm_across_depth(24, 32, 4, constrained, 0.5, 0)
        name = "mHC (Sinkhorn-Knopp)" if constrained else "plain HC          "
        print(f"  {name}: stream RMS at depth 1/8/16/24 = {r['rms'][0]:.2f} / {r['rms'][7]:.2f} / {r['rms'][15]:.2f} / {r['rms'][23]:.2f}"
              f" | max spectral norm of B = {max(r['spec']):.4f}")

    print("\n=== Part 2: hybrid Newton-Schulz and the Muon update ===")
    torch.manual_seed(1)
    M = torch.randn(32, 128)
    sv_fast = torch.linalg.svdvals(hybrid_newton_schulz(M, fast_steps=10, stable_steps=0))
    sv_hyb = torch.linalg.svdvals(hybrid_newton_schulz(M))
    print(f"  singular values after 10 fast steps: min {sv_fast.min():.4f} max {sv_fast.max():.4f} | after 8 fast + 2 stable: min {sv_hyb.min():.4f} max {sv_hyb.max():.4f}")
    for shape in ((32, 128), (96, 32), (4, 32, 64)):
        g = torch.randn(*shape)
        O = muon_update(g, torch.zeros_like(g))
        print(f"  muon_update on {shape}: update RMS {float(O.pow(2).mean().sqrt()):.4f} (target 0.18), max |entry| {float(O.abs().max()):.3f}")

    print("\n=== Part 3: routing balance and SwiGLU clamping ===")
    torch.manual_seed(2)
    scores = torch.sqrt(F.softplus(torch.randn(512, 32) @ torch.randn(4, 32).t() + torch.tensor([2.0, 0.5, 0.0, -2.0])))
    bias = torch.zeros(4)
    load0 = torch.bincount(route_tokens(scores, bias, 2)[0].flatten(), minlength=4)
    for _ in range(300):
        idx, _ = route_tokens(scores, bias, 2)
        bias = update_routing_bias(bias, idx, 4, speed=0.001)
    load1 = torch.bincount(route_tokens(scores, bias, 2)[0].flatten(), minlength=4)
    print(f"  skewed router, 512 tokens x top-2: loads without bias {load0.tolist()} -> after 300 bias updates at speed 0.001 {load1.tolist()}, bias {[round(b, 3) for b in bias.tolist()]}")
    x = torch.randn(1, 8) * 8
    wg, wu, wd = torch.randn(8, 16), torch.randn(8, 16), torch.randn(16, 8) / 4
    print(f"  outlier token (|x| ~ 8): max |SwiGLU output| unclamped {float(clamped_swiglu(x, wg, wu, wd, limit=float('inf')).abs().max()):.1f} vs clamped {float(clamped_swiglu(x, wg, wu, wd).abs().max()):.1f}")

    print("\n=== Part 4: fake quantization error on a (64, 64) Gaussian weight ===")
    torch.manual_seed(3)
    w = torch.randn(64, 64)
    print(f"  MXFP4 (E2M1, 1x32 power-of-two scales): relative error {float((mxfp4_fake_quant(w) - w).norm() / w.norm()):.4f}"
          f" | FP8 E4M3 (128-element scales): {float((fp8_fake_quant(w) - w).norm() / w.norm()):.4f}")

    print("\n=== Part 5: two seeded runs of the tiny V4-style decoder (d_model 32, 2 layers, 4 experts top-2, n_hc 4) ===")
    base = dict(vocab=64, d_model=32, n_layers=2, n_heads=4, d_ff=64, n_experts=4, k=2, n_hc=4, hash_layers=1,
                sinkhorn_iters=10, constrained=True, clamp=True, mtp=True)
    corpus = synthetic_tokens(64, 320, 32, 0)
    data, held = corpus[:256], corpus[256:]
    n_steps, lr = 60, 3e-3
    print("  (Sinkhorn-Knopp runs 10 iterations per block here instead of the report's 20 to keep the CPU run short; for the near-identity maps that occur here the column sums are within 1e-6 of one either way)")
    print(f"  data: 256 training + 64 held-out sequences of 32 tokens from one 4-successor chain over 64 tokens (best possible loss {torch.log(torch.tensor(4.0)):.3f}, uniform {torch.log(torch.tensor(64.0)):.3f})")
    runs = {}
    for name, cfg, kw in (("OFF (plain HC, no clamp, no bias balance, synchronous routing)", dict(base, constrained=False, clamp=False), dict(bias_balance=False, delay=0)),
                          ("ON  (mHC, clamp, bias balance, anticipatory routing dt=4)   ", base, dict(bias_balance=True, delay=4))):
        t1 = time.time()
        res = train_tiny_v4(cfg, data, n_steps, lr, 0, **kw)
        runs[name] = res
        L = res["losses"]
        curve = " ".join(f"{L[i]:.2f}" for i in range(0, n_steps, 10)) + f" {L[-1]:.2f}"
        hold = eval_loss(res["model"], held)
        print(f"\n  run {name}  [{time.time() - t1:.0f}s]")
        print(f"    loss every 10 steps: {curve}")
        print(f"    final loss (mean of last 10) {sum(L[-10:]) / 10:.3f} | held-out loss {hold:.3f} | loss spikes (>1.5x median of previous 20): {res['spikes']}")
        mr, ar = res["muon_rms"], res["adamw_rms"]
        print(f"    Muon update RMS mean {sum(mr) / len(mr):.4f} min {min(mr):.4f} max {max(mr):.4f} | AdamW update RMS first {ar[0]:.3f} mean {sum(ar) / len(ar):.3f} last {ar[-1]:.3f}")
        moes = [b.layer for b in res["model"].blocks if isinstance(b.layer, MoELayer)]
        print(f"    expert loads at the last step: hash layer {res['loads'][0]}, learned layer {res['loads'][1]} | learned-layer bias {[round(b, 3) for b in moes[1].bias.tolist()]}")
        cd = clamp_diagnostics(res["model"], held)
        print(f"    SwiGLU linear pre-activations on held-out data (max |value|, fraction beyond 10): hash layer {cd[0]}, learned layer {cd[1]}")
        print(f"    largest spectral norm of a residual map B after training: {spectral_diagnostics(res['model'], held)}")
        if res["agreement"]:
            ag = res["agreement"]
            print(f"    anticipatory routing agreement with the live router: mean {sum(ag) / len(ag):.3f} min {min(ag):.3f} (fraction of expert choices the stale plan shares)")
    off, on = runs[list(runs)[0]], runs[list(runs)[1]]
    print(f"\n  spikes OFF {off['spikes']} vs ON {on['spikes']}; final loss OFF {sum(off['losses'][-10:]) / 10:.3f} vs ON {sum(on['losses'][-10:]) / 10:.3f}")
    if off["spikes"] == 0 and on["spikes"] == 0:
        print("  (at this scale neither run spikes; the lines above are the diagnostics each mechanism exposes rather than a claimed stability gain)")
    else:
        print("  (spike counts differ between the runs; read them together with the diagnostics above, one seed is one sample)")

    print("\n=== FP4 quantization-aware fine-tuning of the ON model ===")
    model = on["model"]
    fp32 = eval_loss(model, held)
    q0 = eval_loss(model, held, quantized=True)
    qat = train_tiny_v4(base, data, 15, lr / 3, 1, bias_balance=True, delay=0, qat=True, model=model)
    q1 = eval_loss(model, held, quantized=True)
    print(f"  held-out loss: FP32 weights {fp32:.3f} | MXFP4 experts + FP8 dense, no QAT {q0:.3f} ({q0 - fp32:+.3f}) | after 15 QAT steps with straight-through gradients (training continues) {q1:.3f} ({q1 - fp32:+.3f} vs FP32)")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

