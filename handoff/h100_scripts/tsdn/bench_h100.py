"""TSDN cute vs CUB (xllm_extension) on the current GPU: GPU time per call (torch.profiler), % of measured copy SOL.

usage: python bench_h100.py [fwd|bwd|both] [cfg]   (cfg: tuple passed to the cute fwd/bwd, default None = built-in)
"""
import sys
import types
import torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
import xllm_extension.ops as ops
from xllm.modules.fused_ops.cute.timestep_decay_norm import timestep_decay_norm_fwd, timestep_decay_norm_bwd
from torch.profiler import profile, ProfilerActivity


def gpu_time(fn, iters=20):
    """Mean GPU time per call in us (sum of kernel times, so launch gaps do not count)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / iters

SHAPES = [  # B, L, H, G
    (2, 16384, 1536, 24),   # gekko-0.9B
    (1, 8192, 2560, 40),    # gekko-4B
    (1, 16384, 4096, 64),   # gekko-7.1B
    (8, 8192, 2048, 32),    # B200 reference shape
    (8, 8192, 2048, 128),
    (1, 65536, 4096, 64),
]
mode = sys.argv[1] if len(sys.argv) > 1 else "both"
cfg = eval(sys.argv[2]) if len(sys.argv) > 2 else None
dt = torch.bfloat16


def copy_bw():
    a = torch.empty(2 ** 30, dtype=torch.uint8, device="cuda"); b = torch.empty_like(a)
    t = gpu_time(lambda: b.copy_(a))
    return 2 * a.numel() / t / 1e6  # TB/s


PEAK = copy_bw()
print(f"{torch.cuda.get_device_name()}: measured copy bandwidth {PEAK:.2f} TB/s; cfg={cfg}")
print(f"{'':4}{'B':>3} {'L':>6} {'H':>5} {'G':>4} | {'cub us':>8} {'cute us':>8} {'SOL us':>7} {'%SOL':>6} {'speedup':>8}")
for B, L, H, G in SHAPES:
    x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
    pc = torch.zeros(B, dtype=torch.int64, device="cuda")
    pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
    w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
    stats = B * G * L * 8 + B * L * 4
    if mode in ("fwd", "both"):
        tc = gpu_time(lambda: ops.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5))
        tn = gpu_time(lambda: timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5, cfg))
        sol = (2 * x.numel() * 2 + stats) / PEAK / 1e6
        print(f"fwd {B:>3} {L:>6} {H:>5} {G:>4} | {tc:8.1f} {tn:8.1f} {sol:7.1f} {100 * sol / tn:6.1f} {tc / tn:7.1f}x", flush=True)
    if mode in ("bwd", "both"):
        dy = torch.randn_like(x); mg = torch.zeros(B, G, dtype=dt, device="cuda"); vg = torch.zeros(B, G, dtype=dt, device="cuda")
        out = ops.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
        tc = gpu_time(lambda: ops.group_timestep_decay_norm_cub_bwd(dy, mg, vg, x, pc, bos, out[4], out[5], w, None, G, 0.999, 0.9999), 3)
        y, cnt, M, V, cm, cr, sc = timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
        tn = gpu_time(lambda: timestep_decay_norm_bwd(dy, mg, vg, x, bos, cm, cr, sc, w, None, G, 0.999, 0.9999, cfg))
        sol = (3 * x.numel() * 2 + stats) / PEAK / 1e6
        print(f"bwd {B:>3} {L:>6} {H:>5} {G:>4} | {tc:8.1f} {tn:8.1f} {sol:7.1f} {100 * sol / tn:6.1f} {tc / tn:7.1f}x", flush=True)
        del dy, out, y, cm, cr
    del x
    torch.cuda.empty_cache()
