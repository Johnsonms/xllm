# Gekko training on H200: plan for a Claude Code session on an H200 node

H200 is the same sm_90a chip as H100 (same instructions, SM count, tensor throughput and shared memory), with
more and faster HBM. So **nothing has to be ported**: every kernel on the branch
runs as is. The work is (a) reproduce the H100 state and measure it, (b) re-tune what depends on the
memory-bandwidth / compute ratio, and (c) spend the extra memory (141 GB instead of 80 GB) on speed: what was
out of memory or traded away on H100. Read `docs/optimizations/H100.md` first (the "How to run fastest on H100" and
"Summary against the baseline" sections at the top, then §3-§6); this plan refers to it as "H100 handoff".

Branch: all three architectures live on one branch, `gekko-optimizations` (github.com/Johnsonms/xllm); the H200
work goes there too, with its report in `docs/optimizations/H200.md` written the same way as the H100 report. Commit
convention: `git -c user.name="Johnson" -c user.email="johnson@together.ai" commit ...` with the `Co-Authored-By:`
trailer as the last line. No force pushes. The harness scripts the reports' numbers come from are in
`docs/optimizations/harness/`.

## 0. What H200 changes

| | H100 SXM (this port) | H200 (check the node) |
|---|---|---|
| arch | sm_90a, 132 SMs, 228 KB smem / SM, 50 MB L2 | same |
| dense bf16 / fp8 tensor | ~0.99 / ~1.98 PFLOP/s | same |
| HBM | 80 GB HBM3, 3.35 TB/s | 141 GB HBM3e, 4.8 TB/s (~1.43x) |
| NVLink | 900 GB/s, NVSwitch | same on HGX H200 |
| power | 700 W; the bf16 step ran at the cap | 700 W on HGX (H200 NVL / PCIe: 600 W): check `nvidia-smi -q -d POWER,CLOCK` |

Consequences, in order of expected size:
1. **Memory-bound kernels get up to ~1.4x faster** (quantisation and fused SwiGLU ran at 2.6-2.7 TB/s, the
   elementwise / norm / TSDN / conv / AWM-combine kernels near copy bandwidth). Compute-bound parts (GEMMs,
   attention main loops) do not change. So the step shifts further towards GEMMs, which makes the fp8 GEMMs and
   any better fp8 GEMM (DeepGEMM) worth relatively more than on H100.
2. **141 GB per GPU removes most of the memory trade-offs** made on H100 (§3 below): recomputation, resharding,
   swift SCA that did not fit, CUDA-graph blocks that ran out of memory, 7.1B at L16384 only with recompute.
3. Communication contention (the FSDP weight-gradient GEMMs ran at ~440 instead of ~800 TFLOP/s under the
   overlapped reduce-scatter on H100, bandwidth contention) should shrink with the faster HBM. Re-measure.
4. Power: on H100, bf16 was power-capped, fp8 was not (655 W at 1850 MHz). Same TDP, so check whether the
   bf16 step is still at the cap; faster memory at the same power may lower the clock under load.

## 1. Environment and H100-state baseline (do this first)

1. Node: `nvidia-smi` (H200 SXM vs NVL, HBM size), `nvidia-smi topo -m`, driver / CUDA, `ncu` availability
   (H100 handoff §1 has the no-sudo `apt-get download && dpkg -x` route and the CUDA_HOME symlink directory).
2. venv like the H100 one: torch (cu13 build used on H100), `nvidia-cutlass-dsl`, `apache-tvm-ffi`,
   `quack-kernels` 0.6.5, xattn 1.1.1 **with the local patch** (`docs/optimizations/patches/
   xattn-1.1.1-local-full.patch`: stage-2 hdim-256 backward, GQA pair K/V reuse with the seqlen_k == seqlen_q
   fix, precomputed dO.O, vectorized cast; rebuild ~3 min per unit with ninja, watch for stale mtimes). Build
   `xllm_extension` (`python setup.py build_ext --inplace`; it carries `ops.fp8_blockwise_gemm`, without which the
   fp8 path falls back to a separate residual add).
3. Tests before any measurement (the H100 set; all must pass unchanged on sm_90a):
   `tests/test_fp8_sm90.py` (34), `tests/test_fp8_fsdp.py` (5), `tests/test_fp8_gekko_block_sm90.py` (6, needs
   `tests/` on PYTHONPATH), `tests/test_fsdp_copy_in.py`, `tests/test_cute_fused_elementwise.py`,
   `tests/test_cute_adaptive_working_memory.py`, `tests/normalization/test_cute_timestep_decay_norm.py`,
   `tests/attn/test_attention_regressions.py`, `tests/attn/test_swift_blocks.py`; the full SCA sweep
   `tests/sca/test_sliding_chunk_attention.py` as a script. Run them once with the smem-poison plugin
   (`~/h100work/poison` on the H100 node; copy it) to catch uninitialised shared-memory reads.
4. Harness: copy `docs/optimizations/harness/` (profile/run1.sh -> fsdp_trace.py with `CFG_<field>`, `FSDP_RESHARD`,
   `FSDP_FP32_RS`, `BALANCE`, `PROF_STEP`, `WITH_STACK`; `run8_*.sh` drivers; `gaps.py`, trace_top / kern_stack /
   kern_origin).
5. Two baseline tables, 1 GPU and 8 GPUs, ms / step, peak GiB, **and tokens / s / GPU** (batch and sequence
   length will change in §3, so ms / step stops being comparable):
   - **Port baseline** (the H100 starting configs, so the H100 and H200 speedups line up): 0.9B B2 L16384 and
     4B B1 L8192 with xattn + `awm_backend=torch` + `timenorm_backend=cub`, 7.1B B1 L8192 swift.
     H100: 865 / 667 / 1003 ms.
   - **hopper-kernels head**, each row of the H100 "How to run fastest" table (bf16 + symm, fp8, best config).
     H100: 578 / 502 / 770 (bf16 + symm), 465-468 / 393-405 / 581-586 (best).
   The ratio head / port baseline on H200 against H100 tells how much of the H100 work carries over.
6. Profile one step of each preset (trace_top.py, kern_stack.py) and rank kernel time on H200: this ranking,
   not the H100 one, sets the order of §4.

## 2. Carry-over checks (no new code expected)

- Each H100 item has an env switch back to its reference (`XLLM_ROPE_TORCH`, `XLLM_SOFTMAX_BWD_TORCH`,
  `XLLM_SWIGLU_TORCH`, `XLLM_GATE_TORCH`, `XLLM_NORM_EXT`, `XLLM_CE_TORCH`, `AWM_SM90_CUBLAS`, `XLLM_CONV_QK=0`,
  `XLLM_CONV_SM90=0`, `XLLM_QUACK_FFN=0`, `XLLM_QUACK_CE=0`, `XLLM_XATTN_DPSUM=0`, `XATTN_PAIR=0`,
  `XLLM_SWIFT_*`, `XLLM_FP8_*`, `XLLM_FSDP_*`). A 60-step 1-GPU 0.9B parity run (all optimized paths vs all
  references) like the H100 handoff §5 confirms the kernels are correct on H200 too.
- Per-kernel re-measurement of the table in "Summary against the baseline": expect the memory-bound rows
  (TSDN, RoPE, block elementwise, norms, CE rows, quantisation) to gain, the attention / GEMM rows not to.
- Triton kernels (fp8 quantisation, fused SwiGLU / gate quantisation, copy-in, transposes) have fixed configs
  tuned on H100 (num_warps 8-16, 128 x 128 tiles). Re-check num_warps / tiles at the new bandwidth; cheap.

## 3. Spend the memory (the main new lever)

Each item: A/B with alternating repeats (run-to-run noise on H100 was +-5-10 ms on 7.1B, +-3 ms on 4B for
12-step means; a single run proves nothing), peak memory, then a soak against bf16 for anything that changes
numerics. Report tokens / s / GPU next to ms / step.

1. **No recomputation**: `XLLM_GEKKO_SAVE_QK_PREP=1` (H100: +11-20 GB, 0.9B 604 -> 581, 7.1B 827 -> 794 on the
   bf16 path), `recompute_conv_qk=False`, `XLLM_FP8_KEEP_X=1`, `XLLM_GEKKO_SAVE_RMX=1`, `FSDP_RESHARD=0`
   (`reshard_after_forward=False`) on every preset at once; on H100 the best configs mixed these per model only
   because of 80 GB.
2. **7.1B at the plan's shape B1 L16384 without `recompute_fc1_out` / `recompute_fc3_out`** (H100: OOM at 76.6 GB
   without, ~1.70 s with the recompute at 65 GB).
3. **swift SCA on 0.9B / 4B** (OOM on H100; it materialises the attention matrix, ~105 GB at 0.9B B2 L16384) vs
   xattn. swift's cost does not depend on segment boundaries the way xattn's backward does (xattn costs ~1.7x per
   layer without EOS, which made ranks straggle), so check both the step time and the rank spread.
4. **CUDA-graph blocks** (`cuda_graph_block=True`, needs `FSDP_RESHARD=0`): OOM on H100 with fp8 on 4B. The
   smaller presets have launch-bound stretches (H100: ~15 ms compute-stream gaps per 4B step).
5. **Less sharding**: 4B fp32 master weights + Adam (~64 GB) now fit one GPU. Try FSDP2 with
   `reshard_after_forward=False` everywhere, then HSDP / pure replication (all-reduce only, no all-gathers) for
   0.9B and 4B; compare with the symm reduce-scatter path (`XLLM_FSDP_SYMM=1 NCCL_CTA_POLICY=2`).
6. **Larger per-rank batch / longer sequences** where the training recipe allows it: fewer, longer kernels and
   less per-step host and communication overhead. Measure tokens / s, check the loss curve at equal tokens.
7. `XLLM_FP8_AG=1` (fp8 all-gather) only saved memory on H100 (-2.6 to -6 GiB, no speed); not needed on H200
   unless a shape above is tight.

## 4. Re-tune for the new bandwidth / compute ratio

1. **GEMMs become the largest share.** On H100 the bf16 GEMMs ran at 755-790 TFLOP/s (76-80% of dense peak) and
   the blockwise fp8 GEMMs at 1.0-1.24 PFLOP/s via cuBLASLt, which offers exactly one algorithm for blockwise
   scaling. Options, in order:
   - **DeepGEMM** (sm90 blockwise fp8 GEMMs, the same 1 x 128 / 128 x 128 recipe): the largest open item; see
     §7 for the decision and how to proceed under each answer.
   - fp8 by default on H200 (`fp8_gemm=True`): the decision left open on H100.
   - Rowwise fp8 + fast accumulation was measured on H100 and dropped (end-to-end ceiling ~1.5%, the GEMMs are
     compute-bound, so H200 does not change that).
2. **Attention.** xattn's backward at D=256 (0.9B) was L2-bound on the fp32 dQ reduce-adds (ncu: L2 83%, DRAM
   10%); HBM speed does not help it. D=128 (4B) ran at ~500 TFLOP/s useful (FA3-class). Expect little change;
   re-profile before spending time. swift (7.1B) is GEMM + softmax kernels: its softmax / mask kernels gain.
3. **Communication.** Re-measure `NCCL_PROTO=Simple`, bf16 reduce-scatter (`FSDP_FP32_RS=0`), symm
   reduce-scatter and forward prefetch (no gain on H100 because the bf16 step was power-capped; may differ).
4. **Data balance** (`BALANCE=1`, EOS-balanced rows): its value depends on xattn's segment sensitivity; keep it
   in the A/B set.

## 5. Where H100 measured no gain (do not repeat without a reason that H200 changes)

Forward all-gather prefetch, FSDP layer groups, moving FSDP copy-out off the compute stream, a direct Triton
launch path, stacking the q/k/v/r GEMMs, Triton / quack fp8 GEMMs (0.55-0.81 PF and ~370 TF vs cuBLAS
1.17-1.28), rowwise fp8 + fast accumulation (~1.5% ceiling), cuBLAS SM carve-out, FA4 SM90 SCA via mask_mod (10-25%
slower than xattn), xattn main-loop restructure (3-4% ceiling). Memory-limited items (swift on 0.9B / 4B, CUDA-graph
blocks, 7.1B L16384 without recompute) are **not** in this list: they are §3.

## 6. Deliverables

- `docs/optimizations/H200.md` with: node / environment, the two baseline tables, the carry-over check, every §3 /
  §4 item with before / after, memory, A/B repeats and parity, and at the top a "How to run fastest on H200"
  table plus a summary against the port baseline (end to end and per kernel), like the H100 handoff.
- One commit per item on `h200-kernels`, microbenchmark + test + harness numbers in the message; soaks (150-200
  steps against bf16) for any default that changes numerics.
- Decisions to put to the owner, not to make alone: DeepGEMM (§7); fp8 / symm reduce-scatter as defaults;
  changing batch size or sequence length of the training recipe; pushing the xattn patches upstream.

## 7. Decision: DeepGEMM (owner, before the session starts)

**Status: decided 2026-10-25 by the owner (Johnson): option A, allow it.** The H200 session may clone, build
and install DeepGEMM (pinned commit, recorded in `HANDOFF_H200.md`) into the H200 venv, and a newer host compiler
if the node needs one, then follow the steps of option A below: microbenchmark first, integrate only behind an env
switch with cuBLASLt as fallback and default until the 8-GPU A/B and a soak say otherwise. If the permission
policy of that session still blocks the build, point it at this record and ask the owner to allow it there; do
not work around the policy.

Why it matters. On H100 the blockwise fp8 GEMMs go through cuBLASLt, which offers exactly one algorithm for
blockwise scaling (probed), at 1.0-1.24 PFLOP/s, i.e. 50-63% of the ~1.98 PFLOP/s fp8 peak; bf16 cuBLAS reaches
76-80% of its peak. Alternatives measured on H100 were worse (Triton blockwise 0.55-0.81 PF, quack fp8 ~370 TF),
and cuBLASLt rejects fast accumulation with blockwise scales. DeepGEMM (deepseek-ai/DeepGEMM, MIT) implements the
same DeepSeek-V3 recipe on sm90 (1 x 128 activation scales, 128 x 128 weight scales, fp32 promotion) with its own
TMA / wgmma kernels, so it is the one candidate for the GEMM share, which grows on H200 (§0, item 1). The gain is
unmeasured: the build never finished on H100.

What happened on H100. Building it from source needed extra system headers (libdwfl, fetched with
`apt-get download`), then failed on GCC 11 (no `<format>`; it wants a newer GCC or clang); the next build attempt
was denied by the session's permission policy as third-party code ("[Code from External]"), and the session
stopped there rather than work around it.

Measured on H100 the same day (after the decision; details in `HANDOFF_H100.md`, "DeepGEMM on H100"): built
with a `std::format` shim, forward / dgrad GEMMs bitwise equal to cuBLASLt and 1.03-1.14x faster at mid shapes,
~1.0x at the largest; its sm90 weight-gradient kernel (1D1D) is 0.63-0.89x of ours, so wgrad stays on cuBLASLt.
Integrated as `XLLM_FP8_DEEP_GEMM=1` (hopper-kernels c499b30) with `XLLM_FP8_DEEP_GEMM_SMS=120` (all 132 SMs
slowed 4B by competing with the overlapped reduce-scatter): 8 x H100 7.1B -1.1 %, 0.9B -1.5 %, 4B -0.9 %. On H200
re-run the A/B (the GEMM share is larger) and the SM cap sweep, then decide whether to make it the default.

Options:
- **A. Allow it (chosen).** Permit cloning / building / installing DeepGEMM into the H200 venv (pinned
  commit, recorded in the handoff), plus a newer host compiler if the node lacks one. Then: (1) microbenchmark
  against `fp8_sm90.mm` at the gekko shapes (the `h100_scripts/fp8/lt_bench.py` shape list; forward X W^T,
  dgrad dY W, wgrad dY^T X: check which operand layouts / scale granularities DeepGEMM accepts for the weight
  gradient, and whether its scale layout matches ours, M-major 1 x 128 and contiguous 128 x 128, or needs a
  transpose); (2) only if it beats cuBLASLt by enough per GEMM, add it as a backend inside `fp8_sm90.mm` behind an
  env switch (cuBLASLt stays the fallback, and the default until the A/B says otherwise; the fused residual
  epilogue must be kept or its loss counted); (3) parity against `quantize_reference` + cuBLASLt, `tests/test_fp8_sm90.py`
  extended to the new backend, 8-GPU A/B with alternating repeats, then a soak. Its runtime JIT compilation
  (kernels compiled on first use per shape) has to be cached and warmed so it does not distort step timing.
- **B. Allow benchmarking only.** Build it in a scratch directory outside the repo and the training venv, run the
  microbenchmark of A(1), report numbers, no integration. Decide A or C on the data.
- **C. Do not use it.** Keep cuBLASLt. The session then skips the fp8 GEMM backend entirely (§5 lists what was
  already tried) and spends the time on §3.
