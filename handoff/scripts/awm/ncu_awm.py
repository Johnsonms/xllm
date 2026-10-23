import sys
import torch
sys.argv += [] if len(sys.argv) > 1 else ["2,8,4,256,256,16384,4096"]
from check_awm_fwd import make, as_blhs_views, cute_awm
B, H, K, S, V, L, C = (int(x) for x in sys.argv[1].split(","))
args = list(make(B, H, K, S, V, L, C, False, True, dtype=torch.bfloat16))
args[:4] = as_blhs_views(*args[:4])
q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
for _ in range(3):
    cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
torch.cuda.synchronize()
