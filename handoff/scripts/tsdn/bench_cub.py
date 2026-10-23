import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cubext/build"))
import tsdn_cub  # noqa: E402

SHAPES = [  # B, L, H, G
    (8, 8192, 2048, 32),
    (8, 8192, 4096, 32),
    (16, 8192, 2048, 32),
    (8, 32768, 2048, 32),
    (4, 65536, 2048, 32),
    (1, 65536, 4096, 64),
    (8, 8192, 2048, 128),
]


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def main():
    dt = torch.bfloat16
    # bandwidth ceiling: read N + write N bytes
    big = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(big)
    t = bench(lambda: dst.copy_(big))
    peak = 2 * big.numel() / t / 1e6  # TB/s
    print(f"copy bandwidth (1 GiB r + 1 GiB w): {peak:.2f} TB/s")
    del big, dst

    print(f"{'B':>3} {'L':>6} {'H':>5} {'G':>4} | {'fwd us':>8} {'SOL us':>7} {'%SOL':>5} | {'bwd us':>8} {'SOL us':>7} {'%SOL':>5}")
    for B, L, H, G in SHAPES:
        x = torch.randn(B, L, H, dtype=dt, device="cuda")
        bos = torch.rand(B, L, device="cuda") < 0.001
        pc = torch.zeros(B, dtype=torch.int64, device="cuda")
        pm = torch.zeros(B, G, dtype=dt, device="cuda")
        pv = torch.zeros(B, G, dtype=dt, device="cuda")
        w = torch.randn(H, dtype=dt, device="cuda")
        b = torch.randn(H, dtype=dt, device="cuda")
        dy = torch.randn_like(x)
        mg = torch.zeros(B, G, dtype=dt, device="cuda")
        vg = torch.zeros(B, G, dtype=dt, device="cuda")
        out = tsdn_cub.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
        cm, cr = out[4], out[5]
        tf = bench(lambda: tsdn_cub.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5))
        tb = bench(lambda: tsdn_cub.group_timestep_decay_norm_cub_bwd(dy, mg, vg, x, pc, bos, cm, cr, w, None, G, 0.999, 0.9999))
        nbytes = x.numel() * x.element_size()
        stats = B * G * L * 4 * 2  # cummean + cumrstd fp32
        sol_f = (2 * nbytes + stats) / peak / 1e6
        sol_b = (3 * nbytes + stats) / peak / 1e6
        print(f"{B:>3} {L:>6} {H:>5} {G:>4} | {tf:8.1f} {sol_f:7.1f} {100*sol_f/tf:5.1f} | {tb:8.1f} {sol_b:7.1f} {100*sol_b/tb:5.1f}")
        del x, dy, out, cm, cr
        torch.cuda.empty_cache()

    # per-kernel breakdown on the main training shape
    B, L, H, G = 8, 8192, 2048, 32
    x = torch.randn(B, L, H, dtype=dt, device="cuda")
    bos = torch.rand(B, L, device="cuda") < 0.001
    pc = torch.zeros(B, dtype=torch.int64, device="cuda")
    pm = torch.zeros(B, G, dtype=dt, device="cuda")
    pv = torch.zeros(B, G, dtype=dt, device="cuda")
    w = torch.randn(H, dtype=dt, device="cuda")
    b = torch.randn(H, dtype=dt, device="cuda")
    dy = torch.randn_like(x)
    out = tsdn_cub.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            tsdn_cub.group_timestep_decay_norm_cub_fwd(x, bos, pc, pm, pv, w, b, None, G, 0.999, 0.9999, 1e-5)
            tsdn_cub.group_timestep_decay_norm_cub_bwd(dy, pm, pv, x, pc, bos, out[4], out[5], w, None, G, 0.999, 0.9999)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))


if __name__ == "__main__":
    main()
