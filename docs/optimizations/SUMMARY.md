# Gekko training optimizations: summary

Branch `gekko-optimizations`. Details, recipes and caveats: [B200.md](B200.md), [H100.md](H100.md),
[H200.md](H200.md); the full kernel-by-kernel table: [KERNELS.md](KERNELS.md).

## Top 5 kernel optimizations

| # | kernel | what changed | B200 (sm_100a) | H100 / H200 (sm_90a) |
|---|---|---|---|---|
| 1 | Sliding chunk attention | swift (materialised attention matrix, cuBLAS) -> CuTe flash-style fwd / bwd kernels (tcgen05, TMEM operands), fp8 dP option for V = 2048 | 0.9B step 1151 -> 590 ms, peak memory 139 -> 40 GB; 7.1B backward **2.2x** per layer; forward at ~95% of the tensor-pipe peak | not ported: swift backward fused (**1.75x** per layer, 7.1B), xattn patches (**1.1-1.2x**) |
| 2 | Timestep decay norm | CUB scan kernels -> CuTe look-back scan, fwd + bwd, ~85% of HBM roofline | **14.6x / 19.9x** per call (fwd / bwd); 120 -> 2.3 ms per 0.9B step | **~20x** per step (46.6 -> 2.3 ms) |
| 3 | Adaptive working memory | torch / cuBLAS -> CuTe chunk GEMMs + recurrence (tcgen05 on B200, wgmma on Hopper), fwd + bwd | forward **6-11x**, backward **9-16x** per step (0.9B bwd 99 -> 6 ms) | forward **7.4x**, chunk backward **4.0x** |
| 4 | Linear GEMMs in fp8 | bf16 cuBLAS -> MXFP8 (B200, e4m3 + e8m0 / 32) or blockwise e4m3 1 x 128 / 128 x 128 (Hopper), operands written by the fused kernels | GEMM time **1.75-1.84x** lower (7.1B 12.9 -> 7.0 ms per layer); step -8 to -27% | **1.5-1.7x** per GEMM; H200 with DeepGEMM another **1.10-1.35x** |
| 5 | Fused elementwise / norm kernels | torch ops -> CuTe: RoPE, q / k prep (rmsnorm + rope + softmax), conv + q / k prep, attention gate, SwiGLU, group LN, cross-entropy rows | 0.9B: 527 -> 315 ms of step time (**1.7x** on that stage) | RoPE **6x**, block elementwise **3.5x**, conv + q / k prep **1.7x / 2.5x**, group LN **1.75x**, CE gradient GEMM **2.4x** |

## Model speed-ups (8 GPUs, FSDP2, ms per training step fwd + bwd, medians)

| preset | B200 | H100 | H200 |
|---|---|---|---|
| gekko-0.9B B2 L16384 | 1258 -> **270** ms (**4.7x**) | 865 -> **465** (**1.86x**) | 1075 -> **444** (**2.42x**) |
| gekko-4B B1 L8192 | 967 -> **250** (**3.9x**) | 667 -> **393-405** (**1.68x**) | 762 -> **419** (**1.82x**) |
| gekko-7.1B (B200: B1 L16384; Hopper: B1 L8192) | 1150 -> **550** (**2.1x**) | 1003 -> **584** (**1.71x**) | 1082 -> **536** (**2.02x**) |
| tokens / s / GPU at best | 121k / 33k / 30k | 70k / 20k / 14k | 74k / 20k / 15k |

- "start": CUB TSDN, torch AWM, swift / upstream xattn attention, torch elementwise, bf16 GEMMs (on B200 measured on
  1 GPU, same per-rank batch). "best": all kernels on, `recompute_conv_qk=False`, `fsdp_layers_per_group` > 1,
  `reshard_after_forward=False`, `fp8_gemm=True`, plus the per-architecture extras in the reports (CUDA graphs for
  B200 4B, `fp8_attn_dp` for B200 7.1B, symmetric collectives on Hopper, DeepGEMM on H200 7.1B).
- Numerics: unit tests against fp32 / fp64 references for every kernel; end to end on the same batches, bf16 items
  within 2e-4 of loss, fp8 within ~1e-3 over 100-200-step soaks on each architecture.
- H200 7.1B at L16384 fits without the FFN recompute (119 GiB): 1048 ms, 15.4k tokens / s / GPU (OOM on H100).
