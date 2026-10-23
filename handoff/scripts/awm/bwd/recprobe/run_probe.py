import os, sys, importlib.util
import torch
dbg = torch.zeros(1 << 20, dtype=torch.int64, device="cuda")
os.environ["DBG_ADDR"] = str(dbg.data_ptr())
sys.path.insert(0, "/home/johnson/xllm/tests")
from test_cute_adaptive_working_memory import awm, make_inputs
_s = importlib.util.spec_from_file_location("cute_awm_bwd_probe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "awm_bwd_probe.py"))
bwd = importlib.util.module_from_spec(_s); _s.loader.exec_module(bwd)
B, H, K, S, V, L, C = (int(x) for x in sys.argv[1].split(","))
q, qk, k, v, _, _, _, _, _, seg, _ = make_inputs(B, H, K, S, V, L, C, False, 0.3)
w = awm._phases12(qk, k, v, seg, None, None, None, None, None, None, C, save=True)
nc = L // C
dMo = torch.randn(B, K, nc, S, V, device="cuda")
for _ in range(3):
    dbg.zero_()
    bwd._phase2_bwd(w, dMo, None, None, K, S, V, C, False, True)
torch.cuda.synchronize()
nct = B * K * (V // 64)
d = dbg[:nct * 64].view(nct, 64).double()
t0 = d[:, 0].min()
rel = (d - t0) / 1000.0
names = {0: "entry"}
nu = nc - 1
for ur in range(nu):
    for e, n in [(1, "upd start"), (7, "tma first blk"), (2, "upd rG committed"), (3, "mma got rG"), (4, "mma issued"), (5, "upd acc full"), (6, "upd G updated")]:
        names[1 + ur * 8 + e] = f"u{nu - 1 - ur} {n}"
names.update({40: "tma exit", 41: "mma exit", 42: "upd exit"})
for idx in sorted(names):
    col = rel[:, idx]
    if (d[:, idx] == 0).all():
        continue
    print(f"{names[idx]:22s} median {col.median():7.2f}  min {col.min():7.2f}  max {col.max():7.2f} us")
