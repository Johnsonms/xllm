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
| CE grad_input GEMM from the bf16 scores (was tf32 on fp32 scores + fp32 weight copy; same error after the bf16 rounding of grad_input) | 474ec48 | 581 -> 246 us per chunk; step 570 -> 562 ms |
| FFN residual gradient added in the layer norm backward's epilogue | d8c8404 | step 562.2 -> 560.9 ms |
| SwiGLU FFN through quack's gated GEMMs (`gemm_gated` fwd over [w1; w3], `gemm_dgated` for the fc2 input gradient; optional, used when `quack-kernels` is installed, `XLLM_QUACK_FFN=0` off) | 8e93fb7 | FFN fwd+bwd/layer 0.9B 5613 -> 5148 us, 4B 5209 -> 4889, 7.1B equal; step 561 -> 542 ms; 8 GPUs 4B 536 -> 529 |
| fused linear cross entropy through quack's scaled-exp path on sm_90 (scores never written, dw in fp32; `XLLM_QUACK_CE=0` off) | c59b2cc | unchanged at the harness vocab (32000); with the llama3 vocab (128256) step 606 -> 585 ms |
| AWM reverse recurrence on warp MMAs (`AwmRecurrenceBwdWarp`) | d713534 | replaces the torch recurrence backward |
| xattn varlen hdim-256 backward with the 2-stage pipeline (xattn patch, below) | not in xllm | xattn EOS bwd 4.8 -> 3.9 ms/layer; step 585 -> 570 ms (12-step means) |

`quack-kernels` 0.6.5 (Dao-AILab, Apache-2.0, CuTe DSL) is already in this venv from PyPI; the two quack paths are
imported lazily and fall back to the previous code when it is missing. The harness takes `VOCAB=<n>` (default 32000,
the B200 harness value); the llama3 tokenizer in `xllm/data` has 128256 entries, where cross entropy is ~4x larger.

All fused kernels have torch / extension fallbacks behind env switches (`XLLM_ROPE_TORCH`, `XLLM_SOFTMAX_BWD_TORCH`,
`XLLM_SWIGLU_TORCH`, `XLLM_GATE_TORCH`, `XLLM_NORM_EXT`, `XLLM_CE_TORCH`, `AWM_SM90_CUBLAS`) and tests in
`tests/test_cute_fused_elementwise.py` (33) and `tests/test_cute_adaptive_working_memory.py` (20).

Evaluated and not adopted: FA4 SM90 (CuTe) SCA via `mask_mod` + block sparsity (correct, 10-25% slower than xattn);
xattn `segment_idx` route (slower than its BOS route); fewer out-GEMM stages (slower).

### swift SCA (gekko-7.1B, V = 2048; per layer at B1 L8192, fwd + bwd)

| Item | Commit | Effect |
|---|---|---|
| chunk-major saved attention matrices | eec3021 | no strided copies |
| interval-mask softmax kernel (binary search for the segment's key range, 16 B vectors, 256-thread rows) + in-place attention matrix | 1afb985 | fwd 2128 -> 1577 us |
| backward writes q / k / v gradients in place, window overlap accumulated in the GEMMs (`attention_bwd_out`, beta = 1) | 03c6156 | bwd 2929 -> 2765 us, bitwise equal |
| query sub-chunks (`XLLM_SWIFT_QSPLIT`, default 2) skip the masked causal half; output written in place | a94bda8 | fwd+bwd 4261 -> 3827 us; dk / dv rel err vs fp32 4.5e-3 -> 4.7e-3 |
| key ranges start at the sub-chunk's segment start (host copy of the segment ids once per step, cached for all layers and the backward; split 2 or 4 by cost) | 5ab2ae4 | EOS 1/3000: 3796 -> 2279 us, 1/1000: 3775 -> 1648; no EOS unchanged |
| strided batch over heads instead of pointer arrays (B = 1) | 40d6ef9 | 2279 -> 2174 us (EOS 1/3000), 3930 -> 3870 (no EOS) |

8 GPUs, gekko-7.1B B1 L8192: ~940 -> ~847 ms over these items (same batches on all ranks: ~823), peak 51.8 -> 49.4
GiB, losses within 2e-4. Per-rank batches gain less than rank-0 compute (832 -> 786 ms for the segment item): the step
waits for the rank with the fewest EOS, whose attention skips the least. `tests/attn/test_swift_blocks.py` checks the
call layouts against a dense fp32 reference.

### xattn patch (local, `h100_scripts/patches/xattn-1.1.1-varlen-hdim256-bwd-stage2.patch`)

xattn's BOS (EOS-packed) route runs the backward as varlen, and `RunAttentionBwdSm90Kernel` only takes the 2-stage
Q / dO pipeline for non-varlen (`UseDenseStage2 = ... && !kVarlen`), although its shared-memory estimate fits for
D = V = 256. Allowing it for varlen D = V = 256 (one instantiation unit,
`sliding_chunk_attention_bwd_hdim256_vdim256_bf16_sm90.cu`, ~3 min with ninja + relink) takes the 0.9B EOS backward
from ~362 to ~449 TFLOPS (4.8 -> 3.9 ms/layer); dk / dv bitwise equal, dq within 1 bf16 ulp (atomic order). Installed
in this node's venv (original at `~/xattn_cuda.orig.so`); belongs upstream in xattn. 4B (D = 128) already uses 2 stages.

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
Second merge (15ccbff): B200's packed f32x2 rope / softmax math in `cute/qk_prep.py` does not compile below sm_100
("F32x2 intrinsics are not supported"); c0dc6f2 routes it through `_mul2` / `_fma2` (packed on Blackwell, scalar
pairs on Hopper). Worth keeping that pattern upstream for any kernel that has to run on both.

## 4. 8 GPUs (FSDP2, everything above, xattn SCA except 7.1B)

| preset | shape | start of session | now | same batches on all ranks |
|---|---|---|---|---|
| gekko-0.9B | B2 L16384 | swift OOM; xattn + torch AWM 865 ms | 640 ms (xattn patch, CE) | 598 ms (before them) |
| gekko-4B | B1 L8192 | swift OOM; xattn + torch AWM 667 ms | 554 ms | 526 ms |
| gekko-7.1B (swift) | B1 L8192 | 1003 ms (L16384 OOM) | ~847 ms (swift items above) | ~823 ms |

- Straggling (§6c finding) holds on H100 and is larger for 0.9B (14%): xattn's backward costs ~1.7x more per
  layer without segment boundaries, so ranks whose batch has fewer EOS take longer. Data balancing, not kernels.
- Communication is overlapped (4B: 15 ms of compute-stream gaps per step, 12 under NCCL). The collectives slow the
  concurrent GEMMs instead (in isolation the 4B FFN GEMMs run at ~770 TFLOPS). Config, not code (4B, 8 GPUs):
  `NCCL_PROTO=Simple` 577 -> 568 ms; bf16 reduce-scatter (`fp32_reduce_scatter=False`) 568; both 558 (losses equal
  to 4 decimals). NVLS is not available on this node (no algorithm for fp32 reduce-scatter).
  Same on 7.1B: ~883 -> ~878 (Simple) -> ~862 ms (Simple + bf16 reduce-scatter), losses equal to 4 decimals.
- The FFN weight-gradient GEMMs (4B: 9216 x 2560, K 8192) run at ~800 TFLOPS alone but ~440 under the overlapping
  reduce-scatter (memory-bandwidth contention; NCCL already uses RING / SIMPLE for these sizes, the `_RING_LL` in the
  kernel name is NCCL's generic entry point). A cuBLAS SM carve-out (`torch._C._set_sm_carveout_experimental`) does not
  help (8 / 16 / 24 SMs: 554 -> 560-575 ms).
- `reshard_after_forward=False` (keep the unsharded params until the backward) fits on 80 GB and skips the backward
  all-gathers: 4B 554 -> 536 ms (+7 GB), 7.1B ~846 -> ~819 (+12 GB), 0.9B 640 -> 635. Config, not code.
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
  Repeated after the swift / xattn / CE items (reference adds `XLLM_CE_DGRAD_FP32=1` and the unpatched xattn library):
  loss |diff| <= 0.0005, grad norms within 0.24%; median step 322 -> 198 ms.
- 30-step 8-GPU gekko-7.1B B1 L8192, swift items on vs `XLLM_SWIFT_QSPLIT=1 XLLM_SWIFT_SEG_SKIP=0 XLLM_SWIFT_BWD_COPY=1`:
  loss |diff| <= 0.0002, grad norms within 0.03%; median 885 -> 848 ms.
- Comparison script: `h100_scripts/parity_cmp.py <opt.log> <ref.log>`.
- 1 GPU gekko-0.9B B2 L16384 (the plan's shape) fwd+bwd: swift OOM -> 1015 ms (xattn, torch AWM, CUB TSDN) ->
  ~570 ms steady now.
- Also tried, no gain: quack's plain GEMM vs cuBLAS on the projection shapes (wins on some, e.g. K=1024 accumulates
  217 -> 170 us, loses on others); merging the q / k / v / r projections into one GEMM (forward equal, only the
  accumulated input gradient gains ~15%, eaten by the extra copies); Triton conv backward block / warp configs (the
  default is the best of 10); xattn varlen hdim-256 backward with BN 64 / 96 (4490 / 8162 vs 3875 us at BN 80).
- Also tried, no gain (earlier): the B200 CuTe causal conv on H100 (forward ties Triton at ~2.4 TB/s, backward slower); `NCCL_MAX_NCHANNELS` 4 / 8 (4B 554 -> 568-575 ms); xattn with K/V expanded to MHA (bwd -2%,
  more forward work); FLA conv backend (conv 26 -> 44 ms).

## 6. Open items / next

- SCA on 0.9B / 4B is xattn (19-26% of the step). Its BOS route packs segments as varlen sequences: with EOS
  segments the backward launches ~4x the CTAs (13,056 vs 3,280 at 0.9B) and drops from 51% to 38.5% SM throughput.
  A fix belongs in xattn (fixed chunk-aligned tiles with per-row key ranges, as the B200 CuTe SCA does).
- 7.1B SCA (V = 2048) stays on swift (cuBLAS GEMMs + softmax kernels, now ~3.8 ms/layer fwd+bwd). Left: the
  softmax backward is at copy bandwidth; skipping keys before a segment's start needs host-side segment bounds.
  The rest of the 7.1B step is cuBLAS (76-80% of peak) and FSDP2 copies (chunk_cat + split ~44 ms/step).
- xattn hdim-256 backward still carries ~580 us/layer of pre / post kernels (dO.O, dQ cast, GQA dK / dV cast, dQ
  accumulator fill) at 0.9B; xattn-internal.
- AWM: the out GEMM's 7.1B forward epilogue runs at 57% of copy bandwidth (needs a persistent kernel overlapping
  epilogue and loads).
- Remaining small items (0.9B, per step): block elementwise ~34 ms, conv (triton) 26 ms (~1.4x its floor),
  TSDN step_counts recompute 3 ms, FSDP2 split/chunk_cat copies (4B ~19 ms, 7.1B ~21 ms).
