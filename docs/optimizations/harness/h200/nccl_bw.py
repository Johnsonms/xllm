import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); w = dist.get_world_size(); torch.cuda.set_device(r)
for mb in (64, 256, 1024):
    n = mb * 2**20 // 2
    x = torch.randn(n // w, device="cuda", dtype=torch.bfloat16); y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(n, device="cuda", dtype=torch.float32); o = torch.empty(n // w, device="cuda", dtype=torch.float32)
    for _ in range(5): dist.all_gather_into_tensor(y, x); dist.reduce_scatter_tensor(o, z)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20): dist.all_gather_into_tensor(y, x)
    torch.cuda.synchronize(); ag = (time.time() - t) / 20
    t = time.time()
    for _ in range(20): dist.reduce_scatter_tensor(o, z)
    torch.cuda.synchronize(); rs = (time.time() - t) / 20
    if r == 0: print(f"{mb} MB bf16 all-gather {ag*1e3:.2f} ms busbw {mb/1024*(w-1)/w/ag:.0f} GB/s | {mb*2} MB fp32 reduce-scatter {rs*1e3:.2f} ms busbw {2*mb/1024*(w-1)/w/rs:.0f} GB/s", flush=True)
dist.destroy_process_group()
