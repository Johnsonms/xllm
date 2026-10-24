"""Sliding chunk attention through FlashAttention-4's SM90 CuTe kernels (mask_mod + block sparsity) vs xattn.

mask(q, k) = k <= q  and  k >= (q // C - 1) C  and  seg[q] == seg[k]; block masks (128 x 128) from flex's
create_block_mask so that only boundary blocks evaluate the mask and blocks outside the window / segment are skipped.
usage: PYTHONPATH=~/h100work/fa4path python fa4_sca.py 0.9B|4B [eos|none]
"""
import math
import sys
import torch
import cutlass
import cutlass.cute as cute
from torch.profiler import profile, ProfilerActivity
from torch.nn.attention.flex_attention import create_block_mask
from flash_attn.cute import utils
from flash_attn.cute.interface import flash_attn_func
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from xattn.ops import flash_sca_fwd, flash_sca_bwd

SHAPES = {"0.9B": (2, 8, 4, 256, 256), "4B": (1, 32, 8, 128, 128)}
L, C = 16384, 4096
B, H, KV, S, V = SHAPES[sys.argv[1]]
mode = sys.argv[2] if len(sys.argv) > 2 else "eos"
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


@cute.jit
def sca_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    seg = aux_tensors[0]
    m_seg = utils.scalar_to_ssa(seg[batch[0], m_idx[0]], cutlass.Int32)
    n_seg = utils.scalar_to_ssa(seg[batch[0], n_idx[0]], cutlass.Int32)
    c_ssa = utils.scalar_to_ssa(C, cutlass.Int32)
    lo = (m_idx // c_ssa - utils.scalar_to_ssa(1, cutlass.Int32)) * c_ssa
    return (n_idx <= m_idx) & (n_idx >= lo) & (m_seg == n_seg)


g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(B, L, H, S, device="cuda", generator=g).bfloat16()
k = torch.randn(B, L, KV, S, device="cuda", generator=g).bfloat16()
v = torch.randn(B, L, KV, V, device="cuda", generator=g).bfloat16()
dy = torch.randn(B, L, H, V, device="cuda", generator=g).bfloat16()
bos = torch.rand(B, L, device="cuda", generator=g) < (1 / 3000 if mode == "eos" else 0)
bos[:, 0] = True
seg = (torch.cumsum(bos.long(), 1) - 1).to(torch.int32).contiguous()


def flex_mask(b, h, qi, ki):
    return (ki <= qi) & (ki >= (qi // C - 1) * C) & (seg[b, qi] == seg[b, ki])


TN_FWD = int(sys.argv[3]) if len(sys.argv) > 3 else (80 if S == 256 else 128)  # FA4 SM90 forward tile_n
bm_f = create_block_mask(flex_mask, B, H, L, L, device="cuda", BLOCK_SIZE=(128, TN_FWD))
(_, _, kv_cnt, kv_idx, fkv_cnt, fkv_idx, *_) = bm_f.as_tuple()
bs_fwd = BlockSparseTensorsTorch(mask_block_cnt=kv_cnt, mask_block_idx=kv_idx, full_block_cnt=fkv_cnt,
                                 full_block_idx=fkv_idx, block_size=(128, TN_FWD))
TN_BWD = 64 if S == 256 else 128  # FA4 SM90 backward n tile (kv_subtile_factor 1)
bm = create_block_mask(flex_mask, B, H, L, L, device="cuda", BLOCK_SIZE=(128, TN_BWD))
(_, _, _, _, _, _, q_cnt, q_idx, fq_cnt, fq_idx, *_) = bm.as_tuple()
bs_bwd = BlockSparseTensorsTorch(mask_block_cnt=q_cnt, mask_block_idx=q_idx, full_block_cnt=fq_cnt,
                                 full_block_idx=fq_idx, block_size=(128, TN_BWD))
qa, ka, va = (t.detach().requires_grad_(True) for t in (q, k, v))


def fa4_fwd():
    return flash_attn_func(qa, ka, va, softmax_scale=scale, causal=False, mask_mod=sca_mask, aux_tensors=[seg],
                           block_sparse_tensors=bs_fwd, block_sparse_tensors_bwd=bs_bwd, return_lse=True)


out, lse = fa4_fwd()
out.backward(dy)
g4 = (qa.grad.clone(), ka.grad.clone(), va.grad.clone())
y_x, _, lse_x = flash_sca_fwd(q, k, v, C, scale, None, None, bos_mask=bos)
gx = flash_sca_bwd(dy, q, k, v, y_x, lse_x, C, scale, None, None, bos_mask=bos)[:3]
print(f"{sys.argv[1]} {mode}: max|y fa4 - xattn| {(out.float() - y_x.float()).abs().max().item():.3e}; grads",
      [f"{(a.float() - b.float()).abs().max().item():.2e}" for a, b in zip(g4, gx)])
t_fwd = gpu_time(lambda: fa4_fwd())
out, _ = fa4_fwd()
t_fb = gpu_time(lambda: out.backward(dy, retain_graph=True))
tx_f = gpu_time(lambda: flash_sca_fwd(q, k, v, C, scale, None, None, bos_mask=bos))
tx_b = gpu_time(lambda: flash_sca_bwd(dy, q, k, v, y_x, lse_x, C, scale, None, None, bos_mask=bos))
print(f"{sys.argv[1]} {mode}: FA4 fwd {t_fwd:6.0f} us, bwd {t_fb:6.0f} us | xattn fwd {tx_f:6.0f} us, bwd {tx_b:6.0f} us")
