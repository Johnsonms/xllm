"""xattn / swift SCA vs a dense fp32 reference at the Gekko shapes (training mode: bos_mask + segment_idx).

usage: python eval_correct.py <shape> [level]   shape in 0.9B, 4B, 7.1B
Prints max relative errors (max |a - ref| / max |ref|) of y, dq, dk, dv for each backend, with the harness's
eos rate (1/3000) and without segment boundaries.
"""
import sys
import torch
sys.path.insert(0, "/home/johnson/xllm/tests/sca")
from xllm.modules.fused_ops import sliding_chunk_attention_fwd, sliding_chunk_attention_bwd
from xllm.modules.utils import get_bos_mask, get_segment_idx_from_bos_mask
from test_sliding_chunk_attention import mannual_sliding_window_attention

SHAPES = {"0.9B": (2, 8, 4, 256, 256), "4B": (1, 32, 8, 128, 128), "7.1B": (1, 4, 4, 256, 2048)}
L, C = 16384, 4096
name = sys.argv[1]
level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
B, H, KV, S, V = SHAPES[name]
scale = S ** -0.5


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


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
    nseg = int(seg.max()) - int(seg.min()) + 1

    # dense fp32 reference with autograd (scale folded into q)
    qf, kf, vf = (t.float().requires_grad_() for t in (q, k, v))
    yr = mannual_sliding_window_attention(qf * scale, kf, vf, None, None, C, seg, None)
    yr.backward(dy.float())
    ref = (yr.detach(), qf.grad, kf.grad, vf.grad)
    del qf, kf, vf, yr
    torch.cuda.empty_cache()

    for backend in ("xattn", "swift"):
        try:
            y, y_save, aux = sliding_chunk_attention_fwd(q, k, v, C, scale, None, None, bos, seg, 0.0, level, backend, True)
            dq, dk, dv, _, _ = sliding_chunk_attention_bwd(dy, q, k, v, y_save, aux, C, scale, None, None, bos, seg, level, False, backend)
            errs = [rel(a, b) for a, b in zip((y, dq, dk, dv), ref)]
            print(f"{name} {mode:4} (segments/row ~{nseg}) level={level} {backend:5}: rel err y {errs[0]:.2e} dq {errs[1]:.2e} "
                  f"dk {errs[2]:.2e} dv {errs[3]:.2e}", flush=True)
            del y, y_save, aux, dq, dk, dv
        except Exception as e:
            print(f"{name} {mode:4} level={level} {backend:5}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
        torch.cuda.empty_cache()
