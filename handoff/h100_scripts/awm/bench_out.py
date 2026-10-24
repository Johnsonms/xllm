"""AWM output GEMM on Hopper: wgmma kernel (_out_gemm) vs the cuBLAS path, GPU us per call, forward (out = q M) and
backward dq (dOut M^T, S and V swapped). usage: python bench_out.py"""
import sys, types
import torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
from torch.profiler import profile, ProfilerActivity
import importlib
awm = importlib.import_module("xllm.modules.fused_ops.cute.adaptive_working_memory")

SHAPES = [("0.9B", 2, 8, 4, 256, 256), ("4B", 1, 32, 8, 128, 128), ("7.1B", 1, 4, 4, 256, 2048)]
L, C = 16384, 4096
PEAK = 3.04  # TB/s, measured copy bandwidth


def gpu_time(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / iters


for name, B, H, K, S, V in SHAPES:
    nc = L // C
    seg = torch.cumsum((torch.rand(B, L, device="cuda") < 1 / 3000).long(), 1)
    dead = torch.zeros(B, nc, dtype=torch.uint8, device="cuda"); dead[:, 0] = 1
    pc = torch.zeros(1, dtype=torch.int64, device="cuda")
    for role, Sx, Vx, M in (("fwd out", S, V, torch.randn(B, K, nc, V, S, device="cuda").bfloat16()),
                            ("bwd dq ", V, S, torch.randn(B, K, nc, S, V, device="cuda").bfloat16())):
        q = torch.randn(B, L, H, Sx, device="cuda").bfloat16()
        args = (q, M, seg, pc, dead, 0, L, H, K, Sx, Vx, C, False, True)
        o1, m1 = awm._out_gemm(*args)
        o2, m2 = awm._out_gemm_cublas(q, M, seg, pc, dead, 0, H, K, Sx, Vx, C, False, True, torch.empty(B, L, dtype=torch.bool, device="cuda"))
        err = ((o1.float() - o2.float()).abs().max() / o2.float().abs().max()).item()
        assert torch.equal(m1, m2)
        tk = gpu_time(lambda: awm._out_gemm(*args))
        tc = gpu_time(lambda: awm._out_gemm_cublas(q, M, seg, pc, dead, 0, H, K, Sx, Vx, C, False, True, m2))
        nbytes = (q.numel() + o1.numel() + M.numel() * H // K) * 2
        tflops = 2 * B * L * H * Sx * Vx / tk / 1e6
        print(f"{name:5} {role} (K={Sx}, N={Vx}): wgmma {tk:7.1f} us ({tflops:4.0f} TFLOPS, {nbytes / tk / 1e6 / PEAK * 100:3.0f}% of copy BW)"
              f" | cuBLAS path {tc:7.1f} us | {tc / tk:4.1f}x  rel err {err:.1e}", flush=True)
