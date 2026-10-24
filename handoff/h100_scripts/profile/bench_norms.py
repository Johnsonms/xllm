"""xllm_extension group norm ops at the gekko-0.9B B2 L16384 shapes: GPU us per call vs the copy-bandwidth floor."""
import torch
import xllm_extension.ops as ops
from torch.profiler import profile, ProfilerActivity
PEAK = 3.04e12
def gt(fn, it=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(it): fn()
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_time_total > 0]
    return sum(e.device_time_total for e in ev) / it, sorted([(e.key[:50], e.device_time_total / it) for e in ev], key=lambda x: -x[1])
B, L = 2, 16384
for D, G, aff in [(1024, 4, False), (2048, 8, False), (2048, 8, True)]:
    x = torch.randn(B, L, D, device="cuda").bfloat16(); dy = torch.randn_like(x); w = torch.randn(D, device="cuda").bfloat16()
    f = (lambda: ops.group_rms_norm_fwd_affine(x, D, G, w, 1e-6)) if aff else (lambda: ops.group_rms_norm_fwd(x, D, G, 1e-6))
    y, rstd = f()
    b = (lambda: ops.group_rms_norm_bwd_affine(dy, x, D, G, rstd, w, False)) if aff else (lambda: ops.group_rms_norm_bwd(dy, x, D, G, rstd, False))
    nb = x.numel() * 2
    tf, _ = gt(f); tb, kb = gt(b)
    print(f"rms  D{D} G{G} affine={aff}: fwd {tf:6.0f} us (floor {2*nb/PEAK*1e6:5.0f}), bwd {tb:6.0f} us (floor {3*nb/PEAK*1e6:5.0f})  " + "; ".join(f"{n} {t:.0f}" for n, t in kb[:3]))
D, G = 1536, 2
x = torch.randn(B, L, D, device="cuda").bfloat16(); dy = torch.randn_like(x); w = torch.randn(D, device="cuda").bfloat16(); bb = torch.randn(D, device="cuda").bfloat16()
f = lambda: ops.group_layer_norm_fwd_affine(x, D, G, w, bb, 1e-6)
y, mean, rstd = f()
b = lambda: ops.group_layer_norm_bwd_affine(dy, x, D, G, mean, rstd, w, bb, False)
nb = x.numel() * 2
tf, _ = gt(f); tb, kb = gt(b)
print(f"layer D{D} G{G}: fwd {tf:6.0f} us (floor {2*nb/PEAK*1e6:5.0f}), bwd {tb:6.0f} us (floor {3*nb/PEAK*1e6:5.0f})  " + "; ".join(f"{n} {t:.0f}" for n, t in kb[:3]))
