import sys, torch
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
from timestep_decay_norm import timestep_decay_norm_fwd, timestep_decay_norm_bwd
from bench_kernel import gpu_time
dt = torch.bfloat16
B, L, H, G = [int(v) for v in sys.argv[1].split(",")]
x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
pc = torch.zeros(B, dtype=torch.int64, device="cuda"); pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
dy = torch.randn_like(x); mg = torch.zeros(B, G, dtype=dt, device="cuda"); vg = torch.zeros(B, G, dtype=dt, device="cuda")
y, cnt, M, V, cm, cr, sc = timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
sol = (3 * x.numel() * 2 + B * G * L * 8 + B * L * 4) / 6.4 / 1e6
for a in sys.argv[2:]:
    cfg = eval(a)
    try:
        t = gpu_time(lambda: timestep_decay_norm_bwd(dy, mg, vg, x, bos, cm, cr, sc, w, None, G, 0.999, 0.9999, cfg))
        print(f"cfg={cfg}: {t:7.1f} us  {100*sol/t:5.1f}% SOL", flush=True)
    except Exception as e:
        print(f"cfg={cfg}: ERR {[l for l in str(e).splitlines() if l.strip()][:1]}", flush=True)
