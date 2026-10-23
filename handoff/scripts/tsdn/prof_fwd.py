"""TSDN_PROF=1 python prof_bwd.py B,L,H,G : per-tile event timeline of the backward."""
import os, sys, torch
os.environ["TSDN_PROF"] = "1"
sys.path.insert(0, "/home/johnson/xllm/xllm/modules/fused_ops/cute")
import timestep_decay_norm as m
dt = torch.bfloat16
B, L, H, G = [int(v) for v in sys.argv[1].split(",")]
x = torch.randn(B, L, H, dtype=dt, device="cuda"); bos = torch.rand(B, L, device="cuda") < 0.001
pc = torch.zeros(B, dtype=torch.int64, device="cuda"); pm = torch.zeros(B, G, dtype=dt, device="cuda"); pv = torch.zeros(B, G, dtype=dt, device="cuda")
w = torch.randn(H, dtype=dt, device="cuda"); b = torch.randn(H, dtype=dt, device="cuda")
dy = torch.randn_like(x); mg = torch.zeros(B, G, dtype=dt, device="cuda"); vg = torch.zeros(B, G, dtype=dt, device="cuda")
y, cnt, M, V, cm, cr, sc = m.timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
for _ in range(5):
    m.timestep_decay_norm_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5, eval(sys.argv[2]) if len(sys.argv) > 2 else None)
torch.cuda.synchronize()
buf, nt, nch, nseg = m._LAST_PROF_FWD
E = buf[:nt * 8].view(nt, 8).double()
t0 = E[E > 0].min()
E = (E - t0) / 1000.0  # us
names = ["rowsum", "agg0", "agg1", "res0", "res1", "fin0", "fin1", "tma"]
valid = (E[:, 0] > -1)
print(f"tiles {nt} chains {nch} segs {nseg}; span {E[:,[0,4,6]].max():.1f} us")
def st(d, n):
    d = d[torch.isfinite(d)]
    q = torch.quantile(d, torch.tensor([0.1, 0.5, 0.9], dtype=d.dtype, device=d.device))
    print(f"  {n:28s} mean {d.mean():7.2f}  p10 {q[0]:7.2f}  p50 {q[1]:7.2f}  p90 {q[2]:7.2f} us")
st(E[:, 0] - E[:, 7], "tma issue -> moments start")
st(E[:, 1] - E[:, 0], "moments start -> agg start")
st(E[:, 2] - E[:, 1], "aggregate")
st(E[:, 3] - E[:, 2], "agg pub -> resolve start")
st(E[:, 4] - E[:, 3], "resolve (look-back)")
has_fin = E[:, 5] > -1e8
st((E[:, 5] - E[:, 4])[E[:, 6] > 0], "resolve end -> fin call")
st((E[:, 6] - E[:, 5])[E[:, 6] > 0], "finish call (incl. wait)")
st((E[:, 6] - E[:, 0])[E[:, 6] > 0], "moments start -> finish end")
# look-back: time from predecessor's inclusive publish (res1) to ours, along chains
tk = torch.arange(nt, device=E.device)
seg = tk // nch
pred = tk - nch
ok = seg > 0
st((E[ok, 4] - E[pred[ok], 4]), "res1 - pred res1")
st((E[ok, 3] - E[pred[ok], 4]), "res0 - pred res1 (>0: pred ready)")
# per-tile processing rate
print("  per-CTA tile period ~", f"{E[:,0].max() * 148 / nt:.2f} us")
