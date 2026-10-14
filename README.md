# DeepSeek-V4 Training Stack from Scratch

Build the pieces that keep a trillion-parameter MoE training run on the rails, as described in the DeepSeek-V4 technical report, at a size that trains on a laptop CPU. Expand the residual stream into hyper-connection streams and constrain the mixing matrices to the Birkhoff polytope with Sinkhorn-Knopp (mHC), then measure how the signal norm behaves across depth with and without the constraint. Write the Muon optimizer of Algorithm 1: Nesterov momentum, the hybrid Newton-Schulz orthogonalisation, RMS-matched rescaling to 0.18, weight decay, and the split that sends embeddings, heads and norms to AdamW. Build DeepSeekMoE with sqrt-softplus affinities, aux-loss-free bias balancing, hash-routed early layers, shared experts, anticipatory routing and SwiGLU clamping. Fake-quantize expert weights to MXFP4 and the rest to FP8 with straight-through gradients, add the multi-token-prediction module and its loss, and finish by training a tiny V4-style decoder twice, once with the stability mechanisms off and once with them on, printing loss curves, spike counts, routing diagnostics and the optimizer's update RMS.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** RMSNorm
- [x] **2.** expand_residual
- [x] **3.** HCParamGenerator
- [x] **4.** sinkhorn_knopp
- [x] **5.** MHCBlock
- [x] **6.** signal_norm_across_depth
- [x] **7.** newton_schulz_step
- [x] **8.** hybrid_newton_schulz
- [x] **9.** muon_update
- [x] **10.** split_parameters
- [x] **11.** optimizer_step
- [x] **12.** route_tokens
- [x] **13.** update_routing_bias
- [x] **14.** hash_route
- [x] **15.** clamped_swiglu
- [x] **16.** MoELayer
- [x] **17.** anticipatory_routing_plan
- [x] **18.** fp4_e2m1_quantize
- [x] **19.** mxfp4_fake_quant
- [x] **20.** fp8_fake_quant
- [x] **21.** MTPModule
- [x] **22.** mtp_loss
- [x] **23.** CausalAttention
- [x] **24.** TinyV4Decoder
- [x] **25.** synthetic_tokens
- [x] **26.** train_tiny_v4

## Results

```
=== Part 1: signal norm across 24 blocks, static residual maps perturbed by N(0, 0.5) ===
  plain HC          : stream RMS at depth 1/8/16/24 = 0.77 / 3.11 / 44.12 / 479.81 | max spectral norm of B = 2.9297
  mHC (Sinkhorn-Knopp): stream RMS at depth 1/8/16/24 = 1.80 / 8.56 / 16.52 / 24.51 | max spectral norm of B = 1.0000

=== Part 2: hybrid Newton-Schulz and the Muon update ===
  singular values after 10 fast steps: min 0.6818 max 1.1343 | after 8 fast + 2 stable: min 1.0000 max 1.0005
  muon_update on (32, 128): update RMS 0.1800 (target 0.18), max |entry| 0.728
  muon_update on (96, 32): update RMS 0.1800 (target 0.18), max |entry| 0.608
  muon_update on (4, 32, 64): update RMS 0.1800 (target 0.18), max |entry| 0.634

=== Part 3: routing balance and SwiGLU clamping ===
  skewed router, 512 tokens x top-2: loads without bias [326, 249, 261, 188] -> after 300 bias updates at speed 0.001 [256, 257, 255, 256], bias [-0.281, 0.12, 0.11, 0.274]
  outlier token (|x| ~ 8): max |SwiGLU output| unclamped 936.2 vs clamped 130.1

=== Part 4: fake quantization error on a (64, 64) Gaussian weight ===
  MXFP4 (E2M1, 1x32 power-of-two scales): relative error 0.1174 | FP8 E4M3 (128-element scales): 0.0261

=== Part 5: two seeded runs of the tiny V4-style decoder (d_model 32, 2 layers, 4 experts top-2, n_hc 4) ===
  (Sinkhorn-Knopp runs 10 iterations per block here instead of the report's 20 to keep the CPU run short; for the near-identity maps that occur here the column sums are within 1e-6 of one either way)
  data: 256 training + 64 held-out sequences of 32 tokens from one 4-successor chain over 64 tokens (best possible loss 1.386, uniform 4.159)

  run OFF (plain HC, no clamp, no bias balance, synchronous routing)  [2s]
    loss every 10 steps: 4.39 3.90 3.62 3.28 2.97 2.63 2.33
    final loss (mean of last 10) 2.473 | held-out loss 2.319 | loss spikes (>1.5x median of previous 20): 0
    Muon update RMS mean 0.1688 min 0.1673 max 0.1688 | AdamW update RMS first 1.000 mean 0.727 last 0.615
    expert loads at the last step: hash layer [276, 257, 236, 255], learned layer [266, 289, 277, 192] | learned-layer bias [0.0, 0.0, 0.0, 0.0]
    SwiGLU linear pre-activations on held-out data (max |value|, fraction beyond 10): hash layer (5.26, 0.0), learned layer (6.57, 0.0)
    largest spectral norm of a residual map B after training: 1.4935

  run ON  (mHC, clamp, bias balance, anticipatory routing dt=4)     [2s]
    loss every 10 steps: 4.36 3.58 3.11 2.66 2.38 2.04 1.84
    final loss (mean of last 10) 1.954 | held-out loss 1.939 | loss spikes (>1.5x median of previous 20): 0
    Muon update RMS mean 0.1661 min 0.1640 max 0.1668 | AdamW update RMS first 0.999 mean 0.529 last 0.374
    expert loads at the last step: hash layer [276, 257, 236, 255], learned layer [248, 242, 275, 259] | learned-layer bias [0.046, 0.047, -0.06, 0.008]
    SwiGLU linear pre-activations on held-out data (max |value|, fraction beyond 10): hash layer (9.25, 0.0), learned layer (64.38, 0.2231)
    largest spectral norm of a residual map B after training: 1.0
    anticipatory routing agreement with the live router: mean 0.982 min 0.928 (fraction of expert choices the stale plan shares)

  spikes OFF 0 vs ON 0; final loss OFF 2.473 vs ON 1.954
  (at this scale neither run spikes; the lines above are the diagnostics each mechanism exposes rather than a claimed stability gain)

=== FP4 quantization-aware fine-tuning of the ON model ===
  held-out loss: FP32 weights 1.939 | MXFP4 experts + FP8 dense, no QAT 1.968 (+0.029) | after 15 QAT steps with straight-through gradients (training continues) 1.866 (-0.073 vs FP32)

total time 5s
```
