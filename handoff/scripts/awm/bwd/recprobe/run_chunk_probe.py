import os, sys, importlib.util
import torch
dbg = torch.zeros(1 << 20, dtype=torch.int64, device="cuda")
os.environ["DBG_ADDR"] = str(dbg.data_ptr())
sys.path.insert(0, "/home/johnson/xllm/tests")
from test_cute_adaptive_working_memory import awm, make_inputs
_s = importlib.util.spec_from_file_location("cute_awm_chunk_probe", os.path.join(os.path.dirname(os.path.abspath(__file__)), "awm_chunk_probe.py"))
bwd = importlib.util.module_from_spec(_s); _s.loader.exec_module(bwd)
B, H, K, S, V, L, C = (int(x) for x in sys.argv[1].split(","))
q, qk, k, v, _, _, _, _, _, seg, _ = make_inputs(B, H, K, S, V, L, C, False, 0.3)
dout = torch.randn(B, L, H, V, device="cuda").to(torch.bfloat16).transpose(1, 2)
_, _, _, _, ws = awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None, return_workspace=True)
for _ in range(3):
    dbg.zero_()
    state, _ = bwd.awm_bwd_prepare(q, qk, k, v, C, seg, False, ws)
    bwd.awm_bwd_finish(state, dout)
torch.cuda.synchronize()
nct = 148
d = dbg[:nct * 64].view(nct, 64).double()
t0 = d[:, 0].min()
rel = (d - t0) / 1000.0
names = {0: "entry"}
for ti in range(4):
    for e, n in [(0, "epi tile start"), (9, "tma tile issued"), (1, "epi transform done"), (2, "mma dP^T issued"), (11, "mma got P"), (10, "mma last issued"),
                 (3, "epi dk acc full"), (4, "epi dk done"), (5, "epi dqk acc full"), (6, "epi dqk done"), (7, "epi dv acc full"), (8, "epi dv done")]:
        names[1 + ti * 12 + e] = f"t{ti} {n}"
names.update({60: "tma exit", 61: "mma exit", 62: "epi exit"})
for idx in sorted(names):
    col = rel[:, idx]
    if (d[:, idx] == 0).all():
        continue
    print(f"{names[idx]:24s} median {col.median():7.2f}  min {col.min():7.2f}  max {col.max():7.2f} us")
