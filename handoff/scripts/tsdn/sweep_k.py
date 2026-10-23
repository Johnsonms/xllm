import sys, torch
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from timestep_decay_norm import timestep_decay_norm_fwd
from bench_kernel import gpu_time
dt = torch.bfloat16
cfgs = [eval(a) for a in sys.argv[2:]]
B, L, H, G = [int(v) for v in sys.argv[1].split(",")]
x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
pc = torch.zeros(B, dtype=torch.int64, device="cuda"); pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
sol = (2 * x.numel() * 2 + B * G * L * 8 + B * L * 4) / 6.4 / 1e6
for cfg in cfgs:
    try:
        t = gpu_time(lambda: timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5, cfg))
        print(f"B{B} L{L} H{H} G{G} cfg={cfg}: {t:7.1f} us  {100*sol/t:5.1f}% SOL", flush=True)
    except Exception as e:
        print(f"cfg={cfg}: ERR {str(e).splitlines()[0][:100]}", flush=True)
