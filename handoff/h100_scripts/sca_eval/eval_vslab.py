"""7.1B (V = 2048): xattn on 256-wide v slabs (softmax does not depend on V; the backward is linear in the slabs:
dS = sum_s P (dP_s - delta_s)), checked against the fp32 reference and timed. usage: python eval_vslab.py"""
import sys
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, "/home/johnson/xllm/tests/sca")
from xllm.modules.fused_ops import sliding_chunk_attention_fwd, sliding_chunk_attention_bwd
from xllm.modules.utils import get_bos_mask, get_segment_idx_from_bos_mask
from test_sliding_chunk_attention import mannual_sliding_window_attention

B, H, KV, S, V, L, C, VS = 1, 4, 4, 256, 2048, 16384, 4096, 256
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
    t = torch.arange(L, device=seg.device)
    win = torch.clamp((t // C - 1) * C, min=0)
    first = torch.zeros_like(seg)
    first[:, 1:] = (seg[:, 1:] != seg[:, :-1]).long()
    first[:, 0] = 1
    start = torch.cummax(torch.where(first.bool(), t.expand_as(seg), torch.zeros_like(seg)), 1).values
    return (t - torch.maximum(win, start) + 1).sum().item()


def fwd_slabs(q, k, v, bos, seg):
    outs = [sliding_chunk_attention_fwd(q, k, v[..., i:i + VS].contiguous(), C, scale, None, None, bos, seg, 0.0, 0, "xattn", True)
            for i in range(0, V, VS)]
    return torch.cat([o[0] for o in outs], -1), [o[1] for o in outs], outs[0][2]


def bwd_slabs(dy, q, k, v, ys, lse, bos, seg):
    dq = dk = None
    dvs = []
    for j, i in enumerate(range(0, V, VS)):
        a, b, c, _, _ = sliding_chunk_attention_bwd(dy[..., i:i + VS].contiguous(), q, k, v[..., i:i + VS].contiguous(),
                                                    ys[j], lse, C, scale, None, None, bos, seg, 0, False, "xattn")
        dq = a.float() if dq is None else dq + a.float()
        dk = b.float() if dk is None else dk + b.float()
        dvs.append(c)
    return dq.bfloat16(), dk.bfloat16(), torch.cat(dvs, -1)


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
    qf, kf, vf = (t.float().requires_grad_() for t in (q, k, v))
    yr = mannual_sliding_window_attention(qf * scale, kf, vf, None, None, C, seg, None)
    yr.backward(dy.float())
    ref = (yr.detach(), qf.grad, kf.grad, vf.grad)
    del qf, kf, vf, yr
    torch.cuda.empty_cache()
    y, ys, lse = fwd_slabs(q, k, v, bos, seg)
    grads = bwd_slabs(dy, q, k, v, ys, lse, bos, seg)
    errs = [((a.float() - b).abs().max() / b.abs().max()).item() for a, b in zip((y, *grads), ref)]
    tf = gpu_time(lambda: fwd_slabs(q, k, v, bos, seg))
    tb = gpu_time(lambda: bwd_slabs(dy, q, k, v, ys, lse, bos, seg))
    n = pairs(seg) * H
    print(f"7.1B {mode:4} xattn v-slabs x{V // VS}: rel err y {errs[0]:.2e} dq {errs[1]:.2e} dk {errs[2]:.2e} dv {errs[3]:.2e} | "
          f"fwd {tf:6.0f} us ({2 * n * (S + V) / tf / 1e6:4.0f} useful TFLOPS) bwd {tb:6.0f} us "
          f"({2 * n * (3 * S + 2 * V) / tb / 1e6:4.0f}) per layer {tf + tb:6.0f} us", flush=True)
    del ref, y, ys, grads
    torch.cuda.empty_cache()
