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

---

Built on Deep-ML.
