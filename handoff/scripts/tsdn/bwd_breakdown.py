import sys, torch
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from timestep_decay_norm import timestep_decay_norm_fwd, timestep_decay_norm_bwd
from torch.profiler import profile, ProfilerActivity
dt = torch.bfloat16
B, L, H, G = 8, 8192, 2048, 32
x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
pc = torch.zeros(B, dtype=torch.int64, device="cuda"); pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
dy = torch.randn_like(x); mg = torch.zeros(B, G, dtype=dt, device="cuda"); vg = torch.zeros(B, G, dtype=dt, device="cuda")
y, cnt, M, V, cm, cr, sc = timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
f = lambda: timestep_decay_norm_bwd(dy, mg, vg, x, bos, cm, cr, sc, w, None, G, 0.999, 0.9999)
for _ in range(3): f()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10): f()
    torch.cuda.synchronize()
for e in sorted(prof.key_averages(), key=lambda e: -e.device_time_total):
    if e.device_time_total > 0:
        print(f"  {e.key[:70]:70s} {e.device_time_total / 10:8.1f} us/call  x{e.count // 10}")
