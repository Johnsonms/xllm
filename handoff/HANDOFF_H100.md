# H100 (Hopper) port of the Gekko kernel work: hand-off

Working branch `hopper-kernels` (github.com/Johnsonms/xllm), based on `cute-adaptive-working-memory` and
merged with it whenever the B200 session pushes. Plan: `handoff/H100_PLAN.md` on `cute-kernels-handoff`.
Every item below is one commit on `hopper-kernels` with its numbers in the message.

## 1. Node and environment (gpu-dp-mlsf6-dvtr5)

- 8x H100 SXM 80 GB, NV18 all-to-all, driver 610 (CUDA 13.3), 64 CPUs, no system nvcc.
  ncu: `/opt/nvidia/nsight-compute/2026.3.1/ncu` (profiling works without root).
- venv `~/xllm-venv` (uv, python 3.12): torch 2.14.1+cu130, nvidia-cutlass-dsl 4.8.1, apache-tvm-ffi,
  fla-core / flash-linear-attention 0.5.2 (`--no-deps`; 0.6 pulls transformers, whose flash_attn probe breaks
  imports), harness deps of `handoff/venv-freeze.txt`.
- CUDA toolchain for `xllm_extension` / xattn: pip `nvidia-cuda-{nvcc,cccl,crt}==13.0.*`, `nvidia-nvvm==13.0.*`
  (pin nvvm too, or 13.4 comes in), and a CUDA_HOME of symlinks (`bin`, `include`, `nvvm` -> `nvidia/cu13/...`,
  `lib64/*` plus unversioned `libcudart.so`, `libcublas(Lt).so`, `libnvrtc.so`). Build with
  `TORCH_CUDA_ARCH_LIST=9.0`; `setup.py build_ext --inplace` ~3 min.
- xattn v1.1.1 (the version paired with xllm 1.0.2), `~/xattn`: `XATTN_TARGET_SM=90a` build, 640 units, ~80 min.
- The xllm flash backend imports `flash_attn_2_cuda` at import time: scripts stub it
  (`sys.modules.setdefault(...)` or a `flash_attn_2_cuda.py` stub on PYTHONPATH).
- Harness: `handoff/scripts/profile/` copied to `~/h100work/profile`; `run1.sh` was not in the hand-off and is
  recreated (`run1.sh <gpus> <nproc> <port> <trace> <preset> <B> <L> [steps]`); `cats.py` = per-op kernel-time
  table from a `WITH_STACK=1` trace.

## 2. Baseline (H100, before any port)

- 80 GB changes what fits: 1 GPU only gekko-0.9B (B1 L8192 swift: 42.7 GB, 476 ms); gekko-4B OOMs even at
  L4096 (fp32 weights + Adam ~64 GB), gekko-7.1B OOMs creating the Adam state. 4B / 7.1B need FSDP.
- swift SCA on 8 GPUs: 0.9B fits at B1 L16384 (1126 ms, 70 GB), 4B at L4096, 7.1B at L8192 (1176 ms).
- xattn (H100's production SCA, fork evaluation in `~/h100work/sca_eval/`): correct, near the FA3 roofline
  without boundaries (fwd ~790 / bwd ~590 TFLOPS), ~300-450 TFLOPS with EOS segments; head dims <= 256 only
  (7.1B, V = 2048, unsupported). 0.9B B1 L8192: 470 -> ~285 ms, B2 L16384 now fits.

## 3. Ported / new items (step numbers: gekko-0.9B B2 L16384, 1 GPU, xattn SCA, fwd+bwd)

| Item | Commit | Effect |
|---|---|---|
| fused block crash with `awm_backend='torch'` (awm_workspace unbound) | 16153fa | baseline runs at all |
| TSDN cute on Hopper (packed f32x2 -> scalar pairs) | 5571789 | fwd/bwd 11-22x CUB, 77-98% of copy BW; TSDN/step 46.6 -> 2.3 ms |
| packed xattn: segment ids without BOS mask (also broken on main) | b6957ea | attention regressions 6/6 |
| AWM Hopper path (cuBLAS reference, `AWM_SM90_CUBLAS=1`) | 20ecf7c | correctness scaffold |
| AWM wgmma: output GEMM, chunk [A|B], dMo (`adaptive_working_memory_sm90.py`) | 7d7418a | fwd 0.9B 1796 -> 271 us/call |
| AWM wgmma chunk backward, then smem/TMA rework | 8b343b8, dc026fe | chunk bwd 432 -> 109 us; AWM bwd 0.9B 423 us; torch AWM 1015 -> 847 ms step |
| fused RoPE (`cute/rope.py`, any arch) | ccb2a41 | 72 -> 12 ms/step; step 845 -> 787 |
| group norm gamma/beta reductions split over rows (CUDA) | 6306ad8 | step 787 -> 770 |
| fused softmax backward + bf16 softmaxes (`cute/softmax_bwd.py`) | 32a4072 | block elementwise 145 -> 67 ms; step 770 -> 704 |
| fused SwiGLU (`cute/swiglu.py`) | 30c6a60 | step 704 -> 686 |
| group norm forward in CuTe (`cute/group_norm.py`) + tests | 08db9a8 | norm 56 -> 42 ms; step 686 -> 673 |
| fused attention output gate (softplus2 / silu) | ff3fc52 | block elementwise 67 -> 41 ms; step 673 -> 660 |
| fused AdamW by default (`OptimConf.fused`) | 702d594 | optimizer 29 -> 16 ms (1 GPU), 17 -> 13 (8 GPU) |
| CE row kernels (`cute/cross_entropy.py`, MP 1) | 2b4e274 | CE 43 -> 25 ms; step 660 -> 644 |
| group norm backward in CuTe, fused gamma/beta | 652077e | norm 42 -> 32 ms; steady ~630 -> ~620 |

All fused kernels have torch / extension fallbacks behind env switches (`XLLM_ROPE_TORCH`, `XLLM_SOFTMAX_BWD_TORCH`,
`XLLM_SWIGLU_TORCH`, `XLLM_GATE_TORCH`, `XLLM_NORM_EXT`, `XLLM_CE_TORCH`, `AWM_SM90_CUBLAS`) and tests in
`tests/test_cute_fused_elementwise.py` (33) and `tests/test_cute_adaptive_working_memory.py` (20).

Evaluated and not adopted: FA4 SM90 (CuTe) SCA via `mask_mod` + block sparsity (correct, 10-25% slower than xattn);
xattn `segment_idx` route (slower than its BOS route); fewer out-GEMM stages (slower).

### Merge with the B200 fused kernels (bd125c4)

The B200 session added, on `cute-adaptive-working-memory`, fused CuTe kernels overlapping several of the items above:
`cute/rope.py`, `cute/qk_prep.py` (q / k rmsnorm + rope + softmax and their backward), `cute/attn_gate.py` (add +
per-head rmsnorm + gate + multiply), `cute/swiglu.py`, `cute/cross_entropy.py`, `cute/layer_norm.py` and row-split
weight-gradient kernels in the CUDA group norms. `hopper-kernels` merged that branch taking the B200 versions for
these features; their capability gates (`>= 10`) were lowered to sm_90 (no Blackwell-only instructions; all 42 of
their tests pass on H100), which is worth doing upstream too. Kept from this branch on top: AWM Hopper kernels, TSDN
fallback, xattn bos guard, awm_workspace fix, residual adds in the wo / fc2 GEMM epilogues, TSDN step counts handed
to the backward, fused AdamW, swift chunk-major attention matrices; `cute/group_norm.py` / `cute/softmax_bwd.py`
remain with tests but the gekko block now uses the B200 kernels. 0.9B steady step ~596 -> ~570 ms.
Lesson: fetch the B200 branch before starting new fused kernels to avoid duplicate work.

## 4. 8 GPUs (FSDP2, everything above, xattn SCA except 7.1B)

| preset | shape | start of session | now | same batches on all ranks |
|---|---|---|---|---|
| gekko-0.9B | B2 L16384 | swift OOM; xattn + torch AWM 865 ms | 686 ms (after the B200 merge) | 598 ms |
| gekko-4B | B1 L8192 | swift OOM; xattn + torch AWM 667 ms | 554 ms | 526 ms |
| gekko-7.1B (swift) | B1 L8192 | 1003 ms (L16384 OOM) | ~915-945 ms | - |

- Straggling (§6c finding) holds on H100 and is larger for 0.9B (14%): xattn's backward costs ~1.7x more per
  layer without segment boundaries, so ranks whose batch has fewer EOS take longer. Data balancing, not kernels.
- Communication is overlapped (4B: 15 ms of compute-stream gaps per step, 12 under NCCL). The collectives slow the
  concurrent GEMMs instead (in isolation the 4B FFN GEMMs run at ~770 TFLOPS). Config, not code (4B, 8 GPUs):
  `NCCL_PROTO=Simple` 577 -> 568 ms; bf16 reduce-scatter (`fp32_reduce_scatter=False`) 568; both 558 (losses equal
  to 4 decimals). NVLS is not available on this node (no algorithm for fp32 reduce-scatter).
- cuBLAS efficiency (plan §3.6): 7.1B shapes 755-790 TFLOPS (76-80% of dense peak), 4B ~770, 0.9B (K = 1536)
  ~680. Nothing to do kernel-side.

## 5. Validation

- Tests on H100: `tests/test_cute_fused_elementwise.py` 33, `tests/test_cute_adaptive_working_memory.py` 20,
  `tests/normalization/test_cute_timestep_decay_norm.py` 27, `tests/attn/test_attention_regressions.py` 6 pass;
  `tests/test_cute_sliding_chunk_attention.py` skips (Blackwell kernels).
- 60-step training, gekko-0.9B B1 L8192 1 GPU, every optimized path vs every reference path (torch / CUB / extension
  / foreach Adam via the env switches): loss |diff| <= 0.0004, grad norms within 0.3%; median step 324 -> 216 ms.
  Repeated on the merged head (B200 kernels off via `XLLM_CUTE_{CE,GATE,LN,QK,ROPE,SWIGLU}=0`): loss |diff| <= 0.0006,
  grad norms within 0.25%; median step 322 -> 202 ms.
- 1 GPU gekko-0.9B B2 L16384 (the plan's shape) fwd+bwd: swift OOM -> 1015 ms (xattn, torch AWM, CUB TSDN) ->
  ~570 ms steady now.
- Also tried, no gain: `NCCL_MAX_NCHANNELS` 4 / 8 (4B 554 -> 568-575 ms); xattn with K/V expanded to MHA (bwd -2%,
  more forward work); FLA conv backend (conv 26 -> 44 ms).

## 6. Open items / next

- SCA on 0.9B / 4B is xattn (19-26% of the step). Its BOS route packs segments as varlen sequences: with EOS
  segments the backward launches ~4x the CTAs (13,056 vs 3,280 at 0.9B) and drops from 51% to 38.5% SM throughput.
  A fix belongs in xattn (fixed chunk-aligned tiles with per-row key ranges, as the B200 CuTe SCA does).
- 7.1B SCA (V = 2048) stays on swift: FLOP-bound (~12 ms/layer even for an ideal kernel vs swift 10.1 at L8192),
  low ROI on H100. swift's softmax kernels (~39 ms/step) and an output-buffer variant of attention_fwd remain.
- AWM: the backward recurrence is still torch (~100-150 us per call), the out GEMM's 7.1B forward epilogue runs at
  57% of copy bandwidth (needs a persistent kernel overlapping epilogue and loads).
- Remaining small items (0.9B, per step): block elementwise ~34 ms, conv (triton) 26 ms (~1.4x its floor),
  TSDN step_counts recompute 3 ms, FSDP2 split/chunk_cat copies (4B ~19 ms, 7.1B ~21 ms).
