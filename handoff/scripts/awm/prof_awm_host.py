import cProfile, pstats, torch
from check_awm_fwd import make, as_blhs_views, cute_awm
B, H, K, S, V, L, C = 1, 32, 8, 128, 128, 16384, 4096
args = list(make(B, H, K, S, V, L, C, False, True, dtype=torch.bfloat16))
args[:4] = as_blhs_views(*args[:4])
q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
fn = lambda: cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
for _ in range(3):
    fn()
torch.cuda.synchronize()
pr = cProfile.Profile()
pr.enable()
for _ in range(20):
    fn()
torch.cuda.synchronize()
pr.disable()
pstats.Stats(pr).sort_stats("cumulative").print_stats(18)
