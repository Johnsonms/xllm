"""SCA per-call GPU time (torch.profiler kernel sum) and achieved TFLOPS on useful attended pairs.

usage: python bench_sca.py <shape> <backend> [B]     shape in 0.9B, 4B, 7.1B; backend xattn | swift
fwd FLOPs = 2 n (S + V), bwd = 2 n (3 S + 2 V), n = attended (query, key) pairs x heads: per row the keys of the
previous and current chunk up to the row, restricted to the row's segment (eos mode) or not (none mode).
"""
import sys
import torch
from torch.profiler import profile, ProfilerActivity
from xllm.modules.fused_ops import sliding_chunk_attention_fwd, sliding_chunk_attention_bwd
from xllm.modules.utils import get_bos_mask, get_segment_idx_from_bos_mask

SHAPES = {"0.9B": (2, 8, 4, 256, 256), "4B": (1, 32, 8, 128, 128), "7.1B": (1, 4, 4, 256, 2048)}
L, C = 16384, 4096
name, backend = sys.argv[1], sys.argv[2]
B, H, KV, S, V = SHAPES[name]
if len(sys.argv) > 3:
    B = int(sys.argv[3])
scale = S ** -0.5


def gpu_time(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / iters


def pairs(seg):
    """Attended (query, key) pairs per head summed over the batch."""
    t = torch.arange(L, device=seg.device)
    win = torch.clamp((t // C - 1) * C, min=0)  # first key of the window
    first = torch.zeros_like(seg)
    first[:, 1:] = (seg[:, 1:] != seg[:, :-1]).long()
    first[:, 0] = 1
    start = torch.cummax(torch.where(first.bool(), t.expand_as(seg), torch.zeros_like(seg)), 1).values  # segment start
    return (t - torch.maximum(win, start) + 1).sum().item()


for mode in ("eos", "none"):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16()
    k = torch.randn(B, L, KV, S, device="cuda", generator=g).bfloat16()
    v = torch.randn(B, L, KV, V, device="cuda", generator=g).bfloat16()
    dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
    tokens = torch.randint(3, 32000, (B, L), device="cuda", generator=g)
    if mode == "eos":
        tokens[torch.rand(B, L, device="cuda", generator=g) < 1 / 3000] = 2
    bos = get_bos_mask(tokens, 2, use_eos=True)
    seg = get_segment_idx_from_bos_mask(bos)
    n = pairs(seg) * H
    try:
        fwd = lambda: sliding_chunk_attention_fwd(q, k, v, C, scale, None, None, bos, seg, 0.0, 0, backend, True)
        tf = gpu_time(fwd)
        y, y_save, aux = fwd()
        tb = gpu_time(lambda: sliding_chunk_attention_bwd(dy, q, k, v, y_save, aux, C, scale, None, None, bos, seg, 0, False, backend))
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        ff, fb = 2 * n * (S + V), 2 * n * (3 * S + 2 * V)
        print(f"{name} B{B} {mode:4} {backend:5}: fwd {tf:8.0f} us ({ff / tf / 1e6:5.0f} TFLOPS)  bwd {tb:8.0f} us "
              f"({fb / tb / 1e6:5.0f} TFLOPS)  per layer {tf + tb:8.0f} us  peak {peak:.1f} GiB", flush=True)
        del y, y_save, aux
    except Exception as e:
        print(f"{name} B{B} {mode:4} {backend:5}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:140]}", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
