import sys
import torch
from bench_kernel import gpu_time
from check_awm_out import run

for a in sys.argv[1:]:
    B, H, K, S, V, L, C = (int(x) for x in a.split(","))
    compiled, (q, MT, out, mask, ctas) = run(B, H, K, S, V, L, C)
    args = (q.reshape(-1), MT.reshape(-1), out.reshape(-1), mask.view(torch.uint8).reshape(-1), B, L, ctas)
    t = gpu_time(lambda: compiled(*args))
    nbytes = q.numel() * 2 + out.numel() * 2 + MT.numel() * 2 + mask.numel()
    flops = 2 * B * L * H * S * V
    print(f"  {t:7.1f} us   {nbytes / t / 1e6:6.2f} TB/s ({100 * nbytes / t / 1e6 / 8.0:4.1f}% of 8 TB/s)   {flops / t / 1e6:6.1f} TFLOP/s")
