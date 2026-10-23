import time, cProfile, pstats, torch
from check_awm_fwd import make, as_blhs_views, cute_awm
B, H, K, S, V, L, C = 1, 32, 8, 128, 128, 16384, 4096
args = list(make(B, H, K, S, V, L, C, False, True, dtype=torch.bfloat16))
args[:4] = as_blhs_views(*args[:4])
q, qk, k, v, memory, lnt, pqk, pk, pv, seg, cnt = args
fn = lambda: cute_awm.adaptive_working_memory_fwd(q, qk, k, v, C, None, None, None, None, None, seg, None)
for _ in range(5):
    fn()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    fn()
t1 = time.perf_counter()
torch.cuda.synchronize()
print(f"host time per call: {(t1 - t0) / 50 * 1e6:.1f} us")
pr = cProfile.Profile(); pr.enable()
for _ in range(50):
    fn()
pr.disable(); torch.cuda.synchronize()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
