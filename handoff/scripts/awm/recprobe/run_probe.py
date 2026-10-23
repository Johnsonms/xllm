import os, sys
import torch
dbg = torch.zeros(1 << 20, dtype=torch.int64, device="cuda")
os.environ["DBG_ADDR"] = str(dbg.data_ptr())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import awm_probe as cute_awm
from check_awm_fwd import make, as_blhs_views
B, H, K, S, V, L, C = 1, 4, 4, 256, 2048, 16384, 4096
args = list(make(B, H, K, S, V, L, C, False, True, dtype=torch.bfloat16))
args[:4] = as_blhs_views(*args[:4])
q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
for _ in range(3):
    dbg.zero_()
    cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
torch.cuda.synchronize()
d = dbg[:128 * 64].view(128, 64).double()
t0 = d[:, 60].min()
rel = (d - t0) / 1000.0
names = {0: "upd start"}
for c in range(4):
    for e, n in enumerate(["chunk start", "MT written", "m committed", "acc full", "updated"], 1):
        names[1 + c * 8 + (e - 1)] = f"c{c} {n}"
for u in range(3):
    names[40 + u * 4] = f"u{u} mma got M"
    names[41 + u * 4] = f"u{u} mma committed"
names.update({48 + w: f"warp{w} loop done" for w in range(8)})
names.update({56: "mo built", 57: "mem stored", 58: "after barrier"})
names.update({60: "kernel entry", 61: "upd exit", 62: "tma exit", 63: "mma exit"})
for idx in sorted(names):
    col = rel[:, idx]
    if (d[:, idx] == 0).all():
        continue
    print(f"{names[idx]:18s} median {col.median():7.2f}  min {col.min():7.2f}  max {col.max():7.2f} us")
