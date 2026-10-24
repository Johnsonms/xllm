from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from einops import rearrange, repeat
from xllm_extension.ops import (
    attention_fwd,
    attention_bwd,
    multiseg_attention_fwd
)
try:
    from xattn.ops import (
        flash_sca_fwd,
        flash_sca_bwd
    )
    XATTN_SCA_ENABLED = True
except ImportError:
    XATTN_SCA_ENABLED = False
try:
    from ..cute.sliding_chunk_attention import (
        sca_fwd as cute_sca_fwd,
        sca_bwd as cute_sca_bwd,
        segment_starts as cute_segment_starts,
        query_limits as cute_query_limits,
    )
    CUTE_SCA_ENABLED = True
except ImportError:
    CUTE_SCA_ENABLED = False

_CUTE_FALLBACK_WARNED = set()


def cute_sca_supported(q: Tensor, v: Tensor, chunk_size: int, prev_k: Optional[Tensor], dropout: float) -> bool:
    """The CuTe kernels take qk head dims up to 256 (multiples of 64), v head dims that are multiples of 64 up to
    256 or of 256 beyond, chunks of multiples of 128, no previous chunk (context parallel ranks > 0) and no attention
    dropout."""
    S, V = q.shape[3], v.shape[3]
    return (CUTE_SCA_ENABLED and prev_k is None and dropout == 0.0 and S % 64 == 0 and S <= 256 and V % 64 == 0
            and (V <= 256 or V % 256 == 0) and chunk_size % 128 == 0 and q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] >= 10)


def _resolve_backend(backend: str, q: Tensor, v: Tensor, chunk_size: int, prev_k: Optional[Tensor], dropout: float) -> str:
    if backend == 'cute' and not cute_sca_supported(q, v, chunk_size, prev_k, dropout):
        key = (tuple(q.shape[2:]), v.shape[3], chunk_size, prev_k is not None, dropout)
        if key not in _CUTE_FALLBACK_WARNED:
            _CUTE_FALLBACK_WARNED.add(key)
            import warnings
            warnings.warn(f"cute SCA unsupported for head dims {q.shape[3]}/{v.shape[3]}, chunk {chunk_size}, "
                          f"prev_k={prev_k is not None}, dropout={dropout}: falling back to swift", stacklevel=3)
        return 'swift'
    return backend


class SlidingChunkAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_size: int,
        scale: float,
        prev_k: Optional[Tensor] = None,
        prev_v: Optional[Tensor] = None,
        bos_mask: Optional[Tensor] = None,
        segment_idx: Optional[Tensor] = None,
        dropout: float = 0.0,
        high_precision_level: int = 0,
        deterministic: bool = True,
        backend: str = 'swift',
        training: bool = True,
    ) -> Tensor:
        p = dropout if training else 0.0
        y, y_for_save, aux = sliding_chunk_attention_fwd(
            q, k, v, chunk_size, scale, prev_k, prev_v, bos_mask,
            segment_idx, p, high_precision_level, backend, training
        )
        ctx.save_for_backward(
            q, k, v, y_for_save, aux,
            prev_k, prev_v, bos_mask, segment_idx,
        )

        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.dropout = dropout
        ctx.high_precision_level = high_precision_level
        ctx.deterministic = deterministic
        ctx.backend = backend
        return y

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        y_grad: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, None, None,
               Optional[Tensor], Optional[Tensor],
               None, None, None, None, None, None, None]:
        q, k, v, y, aux, prev_k, prev_v, bos_mask, segment_idx = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        scale = ctx.scale
        dropout = ctx.dropout
        high_precision_level = ctx.high_precision_level
        deterministic = ctx.deterministic
        backend = ctx.backend
        q_grad, k_grad, v_grad, prev_k_grad, prev_v_grad = sliding_chunk_attention_bwd(
            y_grad, q, k, v, y, aux, chunk_size, scale, prev_k, prev_v,
            bos_mask, segment_idx, high_precision_level, deterministic, backend
        )

        return q_grad, k_grad, v_grad, None, None, prev_k_grad, prev_v_grad, \
            None, None, None, None, None, None, None


sliding_chunk_attention = SlidingChunkAttentionFunc.apply


def _sequential_sliding_chunk_attention_fwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    chunk_size: int,
    scale: float,
    prev_key_chunk: Optional[Tensor] = None,
    prev_value_chunk: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    prev_segment_idx: Optional[Tensor] = None,
    dropout: float = 0.0,
    save_attention_matrix: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    bsz, seq_len, n_heads, _ = query.shape
    _, _, n_kv_heads, v_head_dim = value.shape
    assert seq_len == key.shape[1] == value.shape[1]
    assert key.shape[2] == n_kv_heads
    assert segment_idx is None or segment_idx.shape[1] == seq_len
    assert prev_segment_idx is None or prev_segment_idx.shape[1] == chunk_size
    assert seq_len % chunk_size == 0

    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads
    # B x L x H x V
    out = torch.empty(bsz, seq_len, n_heads, v_head_dim, dtype=value.dtype, device=value.device)
    if save_attention_matrix:
        # chunk-major: the [B, H, C, keys of chunk c] matrix of every chunk is contiguous in one flat buffer, so the
        # backward reads it without a copy (a [B, H, C, all keys] layout made both passes strided copies)
        offset = 0 if prev_key_chunk is not None else 1
        attn_w = torch.empty(bsz * n_heads * chunk_size * (2 * nc - offset) * chunk_size, dtype=value.dtype,
                             device=value.device)
        w_off = 0
    else:
        offset = None
        attn_w = None
    for c in range(nc):
        qbos = c * chunk_size
        kvbos = c * chunk_size if c == 0 else (c - 1) * chunk_size
        eos = (c + 1) * chunk_size
        q = query[:, qbos:eos]
        k = key[:, kvbos:eos]
        v = value[:, kvbos:eos]
        q_idx = None if segment_idx is None else segment_idx[:, qbos:eos]
        k_idx = None if segment_idx is None else segment_idx[:, kvbos:eos]
        if c == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)
            if k_idx is not None:
                k_idx = torch.cat([prev_segment_idx, k_idx], dim=1)

        # repeat KV
        if n_rep > 1:
            k = repeat(k, 'b l h d -> b l (h n) d', n=n_rep)
            v = repeat(v, 'b l h d -> b l (h n) d', n=n_rep)
        # y: [bsz, chunk_size, nheads, vdim]
        # w: [bsz, nheads, chunk_size, 2*chunk_size]
        if q_idx is None:
            y, w = attention_fwd(q, k, v, scale, dropout, True)
        else:
            y, w = multiseg_attention_fwd(q, k, v, q_idx, k_idx, scale, dropout, True)

        out[:, qbos:eos] = y
        if save_attention_matrix:
            attn_w[w_off:w_off + w.numel()].view(w.shape).copy_(w)
            w_off += w.numel()

    return out, attn_w


def _sequential_sliding_chunk_attention_bwd(
    grad: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    w: Tensor,
    chunk_size: int,
    scale: float,
    prev_key_chunk: Optional[Tensor] = None,
    prev_value_chunk: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    bsz, seq_len, n_heads, v_head_dim = grad.shape
    _, _, n_kv_heads, qk_head_dim = key.shape
    assert seq_len == query.shape[1] and seq_len == key.shape[1] == value.shape[1]
    assert n_heads == query.shape[2] and n_kv_heads == key.shape[2] == value.shape[2]
    assert seq_len % chunk_size == 0
    nc = seq_len // chunk_size
    n_rep = n_heads // n_kv_heads
    # create grad tensors
    query_grad = torch.empty(bsz, seq_len, n_heads, qk_head_dim, dtype=query.dtype, device=query.device)
    key_grad = torch.empty(bsz, seq_len, n_kv_heads, qk_head_dim, dtype=key.dtype, device=key.device)
    value_grad = torch.empty(bsz, seq_len, n_kv_heads, v_head_dim, dtype=value.dtype, device=value.device)
    if prev_key_chunk is not None:
        splits = [2 * chunk_size] * nc
        prev_key_grad = torch.empty(bsz, chunk_size, n_kv_heads, qk_head_dim, dtype=key.dtype, device=key.device)
        prev_value_grad = torch.empty(bsz, chunk_size, n_kv_heads, v_head_dim, dtype=key.dtype, device=key.device)
    else:
        splits = [chunk_size, ] + [2 * chunk_size] * (nc - 1)
        prev_key_grad = None
        prev_value_grad = None

    if w.dim() == 1:  # chunk-major flat buffer of _sequential_sliding_chunk_attention_fwd
        sizes = [bsz * n_heads * chunk_size * kc for kc in splits]
        ws = [t.view(bsz, n_heads, chunk_size, kc) for t, kc in zip(torch.split(w, sizes), splits)]
    else:
        ws = torch.split(w, splits, dim=-1)
    for c in range(nc):
        qbos = c * chunk_size
        kvbos = c * chunk_size if c == 0 else (c - 1) * chunk_size
        eos = (c + 1) * chunk_size
        y_grad = grad[:, qbos:eos]
        q = query[:, qbos:eos]
        k = key[:, kvbos:eos]
        v = value[:, kvbos:eos]
        if c == 0 and prev_key_chunk is not None:
            k = torch.cat([prev_key_chunk, k], dim=1)
            v = torch.cat([prev_value_chunk, v], dim=1)

        # repeat KV
        if n_rep > 1:
            k = repeat(k, 'b l h d -> b l (h n) d', n=n_rep)
            v = repeat(v, 'b l h d -> b l (h n) d', n=n_rep)

        q_grad, k_grad, v_grad = attention_bwd(y_grad, q, k, v, ws[c], scale, True)
        # B x L x H x D
        if n_rep > 1:
            k_grad = rearrange(k_grad, 'b l (h n) d -> b l h n d', n=n_rep).sum(dim=3)
            v_grad = rearrange(v_grad, 'b l (h n) d -> b l h n d', n=n_rep).sum(dim=3)
        # update grad tensors
        query_grad[:, qbos:eos] = q_grad
        curr_k_grad = k_grad[:, -chunk_size:]
        curr_v_grad = v_grad[:, -chunk_size:]
        key_grad[:, qbos:eos] = curr_k_grad
        value_grad[:, qbos:eos] = curr_v_grad

        prev_k_grad = None if k_grad.shape[1] == chunk_size else k_grad[:, :chunk_size]
        prev_v_grad = None if v_grad.shape[1] == chunk_size else v_grad[:, :chunk_size]
        if c == 0:
            prev_key_grad = prev_k_grad
            prev_value_grad = prev_v_grad
        else:
            key_grad[:, kvbos:qbos] += prev_k_grad
            value_grad[:, kvbos:qbos] += prev_v_grad

    return query_grad, key_grad, value_grad, prev_key_grad, prev_value_grad


def sliding_chunk_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    dropout: float = 0.0,
    high_precision_level: int = 0,
    backend: str = 'xattn',
    requires_grad: bool = False
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    backend = _resolve_backend(backend, q, v, chunk_size, prev_k, dropout)
    if backend in ['xattn', 'flash']:
        assert XATTN_SCA_ENABLED, "xattn was not installed."
        assert dropout == 0.0, f"xattn SCA does not support attention dropout: {dropout}"
        if bos_mask is not None:
            segment_idx = None

        y, y_fp32, lse = flash_sca_fwd(
            q, k, v, chunk_size, scale, prev_k, prev_v,
            bos_mask=bos_mask, segment_idx=segment_idx,
            high_precision_level=high_precision_level
        )
        lse = lse if requires_grad else None
        y_for_bwd = (y if high_precision_level == 0 else y_fp32) if requires_grad else None
        return y, y_for_bwd, lse
    elif backend == 'cute':
        assert CUTE_SCA_ENABLED, "the CuTe DSL (nvidia-cutlass-dsl) is not installed."
        assert prev_k is None and prev_v is None, "cute SCA: the previous chunk (context parallel ranks > 0) is not supported."
        assert dropout == 0.0, f"cute SCA does not support attention dropout: {dropout}"
        if segment_idx is None and bos_mask is not None:
            segment_idx = torch.cumsum(bos_mask.long(), dim=1) - 1
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        kmin = cute_segment_starts(segment_idx, q.shape[0], q.shape[1], q.device)
        y, lse2 = cute_sca_fwd(q, k, v, chunk_size, scale, kmin)
        return y, (y if requires_grad else None), (lse2 if requires_grad else None)
    elif backend == 'swift':
        if segment_idx is not None and prev_k is not None:
            assert segment_idx.shape[1] == q.shape[1] + chunk_size
            prev_segment_idx = segment_idx[:, :chunk_size]
            segment_idx = segment_idx[:, chunk_size:]
        else:
            prev_segment_idx = None

        y, w = _sequential_sliding_chunk_attention_fwd(
            q, k, v, chunk_size, scale, prev_k, prev_v, segment_idx, prev_segment_idx, dropout, requires_grad
        )
        return y, None, w
    else:
        raise ValueError(f"Unknown backend: {backend}.")


def sliding_chunk_attention_bwd(
    y_grad: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    y: Optional[Tensor],
    aux: Tensor,  # attn_w for swift & lse for flash
    chunk_size: int,
    scale: float,
    prev_k: Optional[Tensor] = None,
    prev_v: Optional[Tensor] = None,
    bos_mask: Optional[Tensor] = None,
    segment_idx: Optional[Tensor] = None,
    high_precision_level: int = 0,
    deterministic: bool = False,
    backend: str = 'xattn',
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    backend = _resolve_backend(backend, q, v, chunk_size, prev_k, 0.0)
    if backend in ['xattn', 'flash']:
        assert XATTN_SCA_ENABLED, "xattn was not installed."
        if bos_mask is not None:
            segment_idx = None

        return flash_sca_bwd(
            y_grad, q, k, v, y, aux, chunk_size, scale, prev_k, prev_v,
            bos_mask=bos_mask, segment_idx=segment_idx,
            high_precision_level=high_precision_level,
            deterministic=deterministic,
        )
    elif backend == 'cute':
        assert prev_k is None and prev_v is None
        if segment_idx is None and bos_mask is not None:
            segment_idx = torch.cumsum(bos_mask.long(), dim=1) - 1
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        kmin = cute_segment_starts(segment_idx, q.shape[0], q.shape[1], q.device)
        qmax = cute_query_limits(kmin)
        dq, dk, dv = cute_sca_bwd(q, k, v, y, aux, y_grad, chunk_size, scale, kmin, qmax)
        return dq, dk, dv, None, None
    elif backend == 'swift':
        return _sequential_sliding_chunk_attention_bwd(
            y_grad, q, k, v, aux, chunk_size, scale, prev_k, prev_v
        )
    else:
        raise ValueError(f"Unknown backend: {backend}.")
