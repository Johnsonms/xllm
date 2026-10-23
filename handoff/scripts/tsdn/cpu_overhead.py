import sys, time, torch
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from timestep_decay_norm import timestep_decay_norm_fwd
from bench_cub import bench
dt = torch.bfloat16
B, L, H, G = 8, 8192, 2048, 32
x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
pc = torch.zeros(B, dtype=torch.int64, device="cuda"); pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
f = lambda: timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
f(); torch.cuda.synchronize()
n = 50
t = time.perf_counter()
for _ in range(n):
    f()
cpu = (time.perf_counter() - t) / n * 1e6
torch.cuda.synchronize()
print(f"CPU time per call: {cpu:.1f} us; event-timed: {bench(f):.1f} us")
# GPU-only time: profile kernel duration
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        f()
    torch.cuda.synchronize()
for e in prof.key_averages():
    if e.device_time_total > 0:
        print(f"  {e.key[:60]:60s} {e.device_time_total / e.count:8.1f} us x{e.count}")
