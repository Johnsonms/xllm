import sys
import torch
from bench_kernel import gpu_time
from check_awm_ab import run

for a in sys.argv[1:]:
    B, K, S, V, L, C, splits = (int(x) for x in a.split(","))
    fl, fg, (k, qk, v, kmask, lse, AB, nu) = run(B, K, S, V, L, C, splits)
    flat = lambda t: t.reshape(-1)
    tl = gpu_time(lambda: fl(flat(k), flat(kmask), flat(lse), B, L, nu))
    tg = gpu_time(lambda: fg(flat(k), flat(qk), flat(v), flat(kmask), flat(lse), flat(AB), B, L, nu))
    rows = B * nu * C
    nbytes = rows * K * (2 * S + V) * 2 + AB.numel() * 4 * 2 * splits
    flops = 2 * rows * K * S * (S + V)
    print(f"  lse {tl:6.1f} us | gemm {tg:6.1f} us  {nbytes / tg / 1e6:5.2f} TB/s  {flops / tg / 1e6:6.1f} TFLOP/s")
