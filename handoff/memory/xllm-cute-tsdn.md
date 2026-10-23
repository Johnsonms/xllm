---
name: xllm-cute-tsdn
description: "xllm CuTe DSL TimestepDecayNorm kernel project — goal, env setup gotchas, DSL pitfalls, perf lessons, status"
metadata:
  node_type: memory
  type: project
  originSessionId: d4092de1-a881-47cb-aed8-cdcf7765ad74
  modified: 2026-10-22T19:39:50.127Z
---

Task (started 2026-10-22, from Max): write CuTe **Python DSL** kernels for ~/xllm `xllm/modules/timestep_decay_norm.py` (fwd+bwd) first; `adaptive_working_memory.py` later. User wants FA-level performance (near HBM roofline). Machine: 8x B200 (sm_100). Branch `cute-timestep-decay-norm` (pushed to origin = Johnsonms/xllm; user OK'd pushing finished commits), kernel `xllm/modules/fused_ops/cute/timestep_decay_norm.py`, tests `tests/normalization/test_cute_timestep_decay_norm.py` (pytest, loads module by path). No git identity configured: commit with `git -c user.name=Johnson -c user.email=johnson@together.ai`.

**Env gotchas:**
- Use `~/xllm-venv` (py3.10, torch 2.14.1+cu130, nvidia-cutlass-dsl 4.8.1). System python has no torch; no system nvcc; no ncu.
- To build CUDA exts (CUB baseline): pip nvcc/cccl/nvvm/crt must ALL be pinned to 13.0.* to match torch's cu13 headers. Need ninja on PATH, `CPATH=$N/include/cccl:$N/include`, and a `libcudart.so` symlink.
- ptxas from pip nvcc 13.0 can't read DSL PTX (9.4); use a downloaded nvcc 13.4 wheel's ptxas for `-v` register reports.
- ncu: not installed system-wide; `apt-get download nsight-compute-2026.3.1 && dpkg -x` into scratch works without sudo, and profiling works without root. User wants ncu used for bottleneck hunting.
- TVM-FFI (`pip install apache-tvm-ffi`, compile with fake tensors + `--enable-tvm-ffi`) cut host overhead 115us -> ~57us per call.

**CuTe DSL pitfalls (4.8.1):**
- Python-int locals used inside dynamic if/while become dynamic → use `self.X` directly for shapes/range_constexpr.
- Pointer arithmetic involving `self.*` inside *nested* dynamic regions fails ("carried Meta value"); index tensors instead, or use runtime copies (Int32(self.X)) of constants.
- Reusing a variable name across sibling dynamic ifs triggers carry errors; give unique names.
- Ints from select/loops can become signless → `Int32(v).to(Float32)` instead of `Float32(v)`.
- Launch bounds: `.launch(..., min_blocks_per_mp=N)`. Named barriers: `cute.arch.barrier/barrier_arrive(barrier_id=, number_of_threads=)` (max 16 per CTA).
- TMA `cute.copy` elects a lane internally: call it from a FULL warp, never inside `if tidx == 0` (hangs).
- Mixed bf16x2/f32x2 packed ops (fma_packed_f32x2_bf16x2_...) fail to compile on sm_100a; f32x2 packed ops work.
- Python `while` inside jit funcs becomes a dynamic loop; use range_constexpr for compile-time loops.

**Perf lessons (fwd):** fence.acq_rel waits for in-flight prefetch loads → publish look-back state as packed (status|payload) 64-bit relaxed stores, no fences. Decoupled look-back needs aggregates published promptly → separate aggregate warp from resolve warp, and lag normalization behind moments. Watch smem/regs occupancy; compute warps are unsynchronized with each other (caused 2 races).

**Status (2026-10-22, branch head eb3041a, pushed):** fwd + bwd, `timenorm_backend='cute'` wired into TimestepDecayNorm and fused Gekko blocks. bf16 B=8 L=8192 H=2048 G=32 GPU time: fwd 102us (84.5% SOL), bwd 156us (82.3% SOL); CUB 1488/3104us. Bwd 74-81% on main shapes, G=128 ~55%. Results NOT bitwise deterministic (look-back order; dgamma via red.add). Key bwd wins: 1 CTA/SM with T=256 (look-back latency per tile dominated), 2 resolve warps, producer prefetch of per-row stats, dgamma via warp reduce-scatter + red.global.add.v4.f32 from compute warps (no smem staging), bf16 widening via u32 shift/mask. Row sums reduce-scattered (_scatter_rows). Next ideas: keep x/dy in regs across lag.
**ncu workflow:** `CUTE_DSL_LINEINFO=1` + `--import-source yes`, export `--page source --print-source cuda,sass --csv`; rows with Address "-" are per-source-line aggregates incl. stall_* columns (scratch scripts ncu_src.sh / src_an.py).

**How to apply:** Reuse venv/build recipe; compare kernels against SOL, not just CUB.
**Tried, no gain (2026-10-22):** keeping x/dy raw bf16 words in registers across the lag (2x-unrolled loop, ping-pong regs): bwd already at 167 regs/thread (384 thr, 1 CTA/SM) -> spills, 192us (x+dy) / 159us (x only). With setmaxnreg (compute 200 / others 104; 208/88 spills other warps) spills vanish but keep is still not faster; setmaxnreg alone <1%. Smem re-read of x/dy is not the bottleneck.
**Bwd look-back study (2026-10-22, commit after eb3041a):** look-back costs <1% on B=8 L=8192 (nolb bound), ~12% only for few chains (B=1 L=65536 G=32: 176 vs 154us nolb). Per-tile globaltimer probes (scratch edit_prof.py/prof_bwd.py) show with D=1 period >= rowsum->agg->resolve latency - rowsum time; resolve ~2.1us because ~4-5 consecutive segs of a chain are in flight together (inherent with 32 chains). Tried, no gain: lag 2-3 (needs smaller T; also needed mbarrier stage release, which itself cost ~2.5us), starting look-back at TMA landing (worse), nanosleep poll backoff (no effect), dgamma moved into row sums (no effect). Kept: aggregate published + resolve hand-off before per-row q writes (~1% on B=1).
**Fwd look-back study (2026-10-23):** fwd probes: scratch edit_prof_fwd.py + prof_fwd.py. Kept (commit after d20d8ce): NRW=2 resolve warps in fwd + beta^cnt = beta^k*beta^c0 (G128 156->139, H4096 202->188, B8 G32 unchanged ~102). Tried, worse: per-row stats in compute warps (spills at 80 regs w/ 2 CTA/SM; G128 better but others +2us), setmaxnreg split in fwd (more spills, G128 illegal instr), fwd aggregate-first with mbarrier row signal (G128 139->159, agg warp spills). Fwd lag 2 blocked by 16 named barriers. B8 fwd remaining: resolve ~2.3us of which ~1.3 is per-row stats inflated by issue contention; nolb bound 95.6us.
**Fwd lag 2 (2026-10-23):** agg->resolve hand-off now mbarrier per NA slot (freed named barriers, lag 2/3 compile). Lag 2 @1 CTA/SM (1,256,256,1,2,2) hides look-back fully but becomes compute-warp-throughput bound: 100.5 vs 101.7us B8, worse on B1; defaults kept at lag 1. Fwd ~86% HBM SOL, nolb bound 90.5% -> little left from look-back.
