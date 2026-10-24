"""Sweep TSDN cute configs at one shape: python sweep_h100.py fwd|bwd B,L,H,G "cfg1;cfg2;..." (cfg = python tuple)."""
import sys, types
import torch
sys.modules.setdefault("flash_attn_2_cuda", types.ModuleType("flash_attn_2_cuda"))
from torch.profiler import profile, ProfilerActivity
from xllm.modules.fused_ops.cute.timestep_decay_norm import (
    timestep_decay_norm_fwd, timestep_decay_norm_bwd, default_fwd_config, default_bwd_config)


def gpu_time(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / iters


mode, shape, cfgs = sys.argv[1], tuple(int(v) for v in sys.argv[2].split(",")), sys.argv[3]
B, L, H, G = shape
dt = torch.bfloat16
x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
pc = torch.zeros(B, dtype=torch.int64, device="cuda")
pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
ref = timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
dy = torch.randn_like(x); mg = torch.zeros(B, G, dtype=dt, device="cuda"); vg = torch.zeros(B, G, dtype=dt, device="cuda")
y, cnt, M, V, cm, cr, sc = ref
gref = timestep_decay_norm_bwd(dy, mg, vg, x, bos, cm, cr, sc, w, None, G, 0.999, 0.9999)
default = default_fwd_config(dt, H, G) if mode == "fwd" else default_bwd_config(dt, H, G)
for c in [default] + [eval(s) for s in cfgs.split(";") if s.strip()]:
    try:
        if mode == "fwd":
            out = timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5, c)
            err = (out[0].float() - y.float()).abs().max().item()
            t = gpu_time(lambda: timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5, c))
        else:
            out = timestep_decay_norm_bwd(dy, mg, vg, x, bos, cm, cr, sc, w, None, G, 0.999, 0.9999, c)
            err = (out[0].float() - gref[0].float()).abs().max().item()
            t = gpu_time(lambda: timestep_decay_norm_bwd(dy, mg, vg, x, bos, cm, cr, sc, w, None, G, 0.999, 0.9999, c))
        print(f"{mode} {shape} cfg={c}{' (default)' if c == default else ''}: {t:7.1f} us  maxdiff vs default {err:.2e}", flush=True)
    except Exception as e:
        print(f"{mode} {shape} cfg={c}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True)
