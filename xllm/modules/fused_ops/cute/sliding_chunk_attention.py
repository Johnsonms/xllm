"""Sliding chunk attention (SCA) with CuTe DSL kernels for Blackwell (SM100).

Queries of chunk c (rows [cC, (c+1)C)) attend causally to the keys of chunks c - 1 and c (chunk 0 only to itself),
restricted to the query's own segment (segment ids are nondecreasing, so a row's valid keys are the range
[segment start, row]). Nothing is materialized: the forward is a flash-attention style kernel that keeps the
128-row query tile in shared memory, streams 64-key blocks of K / V, runs the online softmax in 4 warps (one row
per thread) and feeds P to the PV MMA from tensor memory. It saves only the output and the row log-sum-exp.

Layouts are those of the fused block: q [B, L, H, S], k [B, L, K, S], v [B, L, K, V] (bf16), y [B, L, H, V].
lse2 [B, H, L] (fp32) is the row log2-sum-exp of the scaled scores (scale * log2 e applied), the form the
backward recomputes P from.
"""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05

try:
    from .adaptive_working_memory import _compiled, _fake, _sm_count, NEG_BIG, LOG2E  # noqa: F401
except ImportError:  # loaded from its file (tests)
    import importlib.util as _ilu
    import os as _os
    _spec = _ilu.spec_from_file_location("cute_adaptive_working_memory_fwd",
                                         _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "adaptive_working_memory.py"))
    _fwd = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fwd)
    _compiled, _fake, _sm_count, NEG_BIG, LOG2E = _fwd._compiled, _fwd._fake, _fwd._sm_count, _fwd.NEG_BIG, _fwd.LOG2E

def _fake2(dtype, align=16):
    """A 2-D fake tensor: element counts beyond the int32 range of the 1-D fake tensors (only its pointer is used)."""
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(), cute.sym_int()), assumed_align=align)


def _col2d(t: torch.Tensor) -> torch.Tensor:
    """A contiguous tensor as the column-major 2-D view the 2-D fake tensors expect (pointer only)."""
    return t.reshape(1, -1).t()


RESCALE_THRESHOLD = 8.0  # log2 units: the running max is only moved (and O rescaled) when it grows by more


class ScaFwd:
    """y = softmax(scale q k^T, masked) v for one (batch, head, chunk, 128-row tile, 256-wide v slab) per CTA.

    Warps 0-3: softmax (thread = row = TMEM lane), O rescale and the epilogue; warp 4: tcgen05 MMAs; warp 5: TMA.
    TMEM: S [128 x KB] fp32 x 2 stages, P [128 x KB] bf16 (A operand of the PV MMA), O [128 x VT] fp32.
    Shared memory: the q tile (128 x S), 2 stages of K (KB x S) and V (KB x VT) blocks. For V > 256 the CTAs of
    the V // 256 slabs of a tile each recompute S (its cost is S / V of the PV MMA)."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, KB: int = 64, kv_stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.KB = H, K, S, V, C, KB
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, 256)
        self.n_vt = V // self.VT
        self.kv_stages = kv_stages
        self.num_softmax_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        self.row_tiles = C // self.TM
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 32 == 0 and V % self.VT == 0 and C % KB == 0 and C % 128 == 0
        self.tmem_s = 0
        self.tmem_p = 2 * KB
        self.tmem_o = 2 * KB + KB // 2
        total = self.tmem_o + self.VT
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tKmin: cute.Tensor, tY: cute.Tensor,
                 tLse: cute.Tensor, B: Int32, L: Int32, scale2: Float32, stream: cuda.CUstream):
        H, K, S, V, KB, VT = self.H, self.K, self.S, self.V, self.KB, self.VT
        nc = L // self.C
        # q as (M = L, K = S, H, B); k as (N = L, K = S, Kh, B); v as (N = V, K = L, Kh, B) (MN-major B of PV)
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((V, L, K, B), stride=(1, K * V, V, L * K * V)))

        tm_qk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_pv = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        q_layout = sm100_utils.make_smem_layout_a(tm_qk, (self.TM, KB, S), BFloat16, 1)
        k_layout = sm100_utils.make_smem_layout_b(tm_qk, (self.TM, KB, S), BFloat16, self.kv_stages)
        v_layout = sm100_utils.make_smem_layout_b(tm_pv, (self.TM, VT, KB), BFloat16, self.kv_stages)
        p_layout = sm100_utils.make_smem_layout_a(tm_pv, (self.TM, VT, KB), BFloat16, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, KB, S), tm_qk)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), (self.TM, KB, S), tm_qk)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, cute.slice_(v_layout, (None, None, None, 0)), (self.TM, VT, KB), tm_pv)
        num_tiles = B * nc * H * self.row_tiles * self.n_vt
        self.kernel(tm_qk, tm_pv, tma_q, mQ_t, tma_k, mK_t, tma_v, mV_t, tKmin.iterator, tY.iterator, tLse.iterator,
                    q_layout, k_layout, v_layout, p_layout, L, nc, scale2).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_qk: cute.TiledMma, tm_pv: cute.TiledMma, tma_q: cute.CopyAtom, mQ: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor, pKmin: cute.Pointer,
               pY: cute.Pointer, pLse: cute.Pointer, q_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout, L: Int32, nc: Int32, scale2: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, KB, C, VT = self.H, self.K, self.S, self.V, self.KB, self.C, self.VT

        # tile -> (v slab, row tile, head, chunk, batch); the slabs of a tile and the heads of a kv group are
        # adjacent (they share the q tile / the K and V blocks)
        vt = bidx % self.n_vt
        t0 = bidx // self.n_vt
        rt = t0 % self.row_tiles
        t1 = t0 // self.row_tiles
        h = t1 % H
        t2 = t1 // H
        c = t2 % nc
        b = t2 // nc
        kh = h // self.n_rep
        q0 = c * C + rt * self.TM  # first query row of the tile
        kstart = Int32(0)
        if c > 0:
            kstart = (c - 1) * C
        kmin0 = cute.make_tensor(pKmin + (b * L + q0), cute.make_layout(1))[0]
        kb_lo = (kmin0 - kstart) // KB
        if kb_lo < 0:
            kb_lo = Int32(0)
        kb_hi = (q0 + self.TM - 1 - kstart) // KB
        n_blocks = kb_hi - kb_lo + 1
        kblk0 = kstart // KB + kb_lo  # first key block (units of KB rows of the sequence)

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)

        @cute.struct
        class SharedStorage:
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, self.kv_stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            p_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_softmax_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        sm_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_softmax_warps)
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))
        v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2]))
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        k_prod, k_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.kv_stages, producer_group=one, consumer_group=one,
            tx_count=k_bytes).make_participants()
        v_prod, v_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.v_mbar.data_ptr(), num_stages=self.kv_stages, producer_group=one, consumer_group=one,
            tx_count=v_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=2, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_softmax_warps)).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mbar.data_ptr(), num_stages=1, producer_group=sm_group, consumer_group=one).make_participants()

        sQ = smem.allocate_tensor(BFloat16, q_layout.outer, byte_alignment=1024, swizzle=q_layout.inner)
        sK = smem.allocate_tensor(BFloat16, k_layout.outer, byte_alignment=1024, swizzle=k_layout.inner)
        sV = smem.allocate_tensor(BFloat16, v_layout.outer, byte_alignment=1024, swizzle=v_layout.inner)

        gQ = cute.local_tile(mQ, (self.TM, S), (None, None, None, None))  # (128, S, L/128, 1, H, B)
        gK = cute.local_tile(mK, (KB, S), (None, None, None, None))  # (KB, S, L/KB, 1, Kh, B)
        gV = cute.local_tile(mV, (VT, KB), (None, None, None, None))  # (VT, KB, V/VT, L/KB, Kh, B)
        thr_qk = tm_qk.get_slice(0)
        thr_pv = tm_pv.get_slice(0)
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_qk.partition_A(gQ), 0, 3))
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(thr_qk.partition_B(gK), 0, 3))
        tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(thr_pv.partition_B(gV), 0, 3))
        tCrQ = tm_qk.make_fragment_A(sQ)
        tCrK = tm_qk.make_fragment_B(sK)
        tCrV = tm_pv.make_fragment_B(sV)
        s_fake = tm_qk.make_fragment_C(cute.append(tm_qk.partition_shape_C((self.TM, KB)), 2))
        o_fake = tm_pv.make_fragment_C(tm_pv.partition_shape_C((self.TM, VT)))
        tP_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), p_layout.outer)
        tPr_fake = tm_pv.make_fragment_A(tP_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            hq = q_prod.acquire_and_advance()
            cute.copy(tma_q, tQg[(None, q0 // self.TM, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
            for j in cutlass.range(n_blocks):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, kblk0 + j, 0, kh, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                hv = v_prod.acquire_and_advance()
                cute.copy(tma_v, tVg[(None, vt, kblk0 + j, kh, b)], tVs[(None, hv.index)], tma_bar_ptr=hv.barrier)
            q_prod.tail()
            k_prod.tail()
            v_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            q_cons.wait()
            # S_0
            es = s_prod.acquire_and_advance()
            hk = k_cons.wait_and_advance()
            tS0 = tS[(None, None, None, es.index)]
            for kk in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
                tm_qk.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_qk, tS0, tCrQ[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tS0)
            hk.release()
            es.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hk = k_cons.wait_and_advance()
                    tSj = tS[(None, None, None, es.index)]
                    for kk in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
                        tm_qk.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_qk, tSj, tCrQ[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tSj)
                    hk.release()
                    es.commit()
                hp = p_cons.wait_and_advance()
                hv = v_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tPr, mode=[2])):
                    tm_pv.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_pv, tO, tPr[(None, None, kk)], tCrV[(None, None, kk, hv.index)], tO)
                hv.release()
                hp.release()
            q_cons.release()
            s_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            trow = tidx % 128
            qpos = q0 + trow
            kmin_row = cute.make_tensor(pKmin + (b * L + qpos), cute.make_layout(1))[0]
            kmin_last = cute.make_tensor(pKmin + (b * L + q0 + self.TM - 1), cute.make_layout(1))[0]
            # S loads: this thread's row of a stage (KB columns)
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(KB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, KB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, KB))).shape, Float32)
            # P store: bf16 pairs into the TMEM A operand of the PV MMA
            st_p = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(KB // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_p = tcgen05.make_tmem_copy(st_p, tPr)
            thr_p = r2t_p.get_slice(trow)
            tRT_tP = thr_p.partition_D(tPr)
            tRT_c = thr_p.partition_S(thr_pv.partition_A(cute.make_identity_tensor((self.TM, KB))))
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            pbf_v = cute.make_tensor(pbf.iterator, cute.make_layout(tRT_c.shape))
            # O pieces of 32 columns: loads (epilogue, rescale) and stores (rescale)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            st_o = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE), Float32)
            tO_r = cute.flat_divide(tO[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tO_r[(None, None, 0, 0)])
            r2t_o = tcgen05.make_tmem_copy(st_o, tO_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tO_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            tRT_o = r2t_o.get_slice(trow).partition_D(tO_r)
            tRT_o = cute.group_modes(tRT_o, 3, cute.rank(tRT_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)

            m_run = Float32(NEG_BIG)
            l_run = Float32(0.0)
            for j in cutlass.range(n_blocks):
                bstart = kstart + (kb_lo + j) * KB
                fs = s_cons.wait_and_advance()
                if fs.index == 0:
                    cute.copy(t2r_s[0], tTR_s[0][(None, None, None, 0)], xs)
                else:
                    cute.copy(t2r_s[1], tTR_s[1][(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                # mask: the valid keys of a row are [kmin_row, qpos]
                if (bstart < kmin_last) | (bstart + KB - 1 > q0):
                    lo = kmin_row - bstart
                    hi = qpos - bstart
                    for e in cutlass.range_constexpr(KB):
                        keep = (e >= lo) & (e <= hi)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))  # exp2 -> 0, never moves the max
                rmax = xs[0]
                for e in cutlass.range_constexpr(1, KB):
                    rmax = cute.arch.fmax(rmax, xs[e])
                m_old = m_run
                m_new = cute.arch.fmax(m_old, rmax * scale2)
                moved = m_new > m_old + RESCALE_THRESHOLD
                if moved:
                    m_run = m_new
                m_use = m_run
                if m_run < -1.0e29:  # no valid key yet: exp2 of the stand-in stays 0
                    m_use = Float32(0.0)
                neg_m = -m_use
                rsum = Float32(0.0)
                for e2 in cutlass.range_constexpr(KB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2), (neg_m, neg_m))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                    rsum = rsum + xs[2 * e2] + xs[2 * e2 + 1]
                pbf.store(xs.load().to(BFloat16))
                # the P buffer is free (and O complete) once the previous PV MMA finished
                hp = p_prod.acquire_and_advance()
                # O rescale: TMEM loads / stores are warp-collective, so the whole warp takes the branch
                corr = Float32(1.0)
                if moved & (j > 0):
                    corr = cute.math.exp2(m_old - m_new, fastmath=True)
                    l_run = l_run * corr
                if cute.arch.vote_any_sync(moved & (j > 0)):
                    for piece in cutlass.range_constexpr(VT // 32):
                        cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                        cute.arch.fence_view_async_tmem_load()
                        ov.store(ov.load() * corr)
                        cute.copy(r2t_o, ov, tRT_o[(None, None, None, piece)])
                    cute.arch.fence_view_async_tmem_store()
                l_run = l_run + rsum
                cute.copy(r2t_p, pbf_v, tRT_tP)
                cute.arch.fence_view_async_tmem_store()
                hp.commit()
            # all PV MMAs done
            hp = p_prod.acquire_and_advance()
            inv_l = Float32(1.0) / l_run
            row_ptr = pY + ((Int64(b * L + qpos) * H + h) * V + vt * VT)
            for piece in cutlass.range_constexpr(VT // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store((ov.load() * inv_l).to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            if vt == 0:
                cute.make_tensor(pLse + ((b * H + h) * L + qpos), cute.make_layout(1))[0] = m_run + cute.math.log2(l_run, fastmath=True)
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_softmax_warps)
            tmem.free(tmem_ptr)


class ScaBwdDv:
    """dv[keys of a 128-row tile] = sum over the attending queries P^T dO, for one (batch, kv head, key tile) per CTA.

    The CTA loops over the heads of the kv group and the 64-query blocks that can attend the tile (from the tile's
    own rows to the last query whose segment reaches the tile, within the next chunk). Per block: S^T = K Q^T
    (tcgen05, K tile resident), P^T = exp2(S^T scale2 - lse2[q]) with the mask (warps 0-3, thread = key row), P^T
    into TMEM, dV += P^T dO (dO block as MN-major B). TMEM: S^T x 2, P^T, dV."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, QB: int = 64, stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.QB = H, K, S, V, C, QB
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, 256)
        self.n_vt = V // self.VT
        self.stages = stages
        self.num_cw = 4  # compute warps
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 32 == 0 and V % self.VT == 0 and C % 128 == 0 and C % QB == 0
        self.tmem_s = 0
        self.tmem_p = 2 * QB
        self.tmem_o = 2 * QB + QB // 2
        total = self.tmem_o + self.VT
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor, tKmin: cute.Tensor,
                 tQmax: cute.Tensor, tDV: cute.Tensor, B: Int32, L: Int32, scale2: Float32, stream: cuda.CUstream):
        H, K, S, V, QB, VT = self.H, self.K, self.S, self.V, self.QB, self.VT
        nc = L // self.C
        # k as (M = L, K = S, Kh, B); q as (N = L, K = S, H, B); dO as (N = V, K = L, H, B) (MN-major B)
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((V, L, H, B), stride=(1, H * V, V, L * H * V)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_o = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, VT, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        a_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, S), BFloat16, 1)
        q_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, S), BFloat16, self.stages)
        do_layout = sm100_utils.make_smem_layout_b(tm_o, (self.TM, VT, QB), BFloat16, self.stages)
        p_layout = sm100_utils.make_smem_layout_a(tm_o, (self.TM, VT, QB), BFloat16, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), (self.TM, VT, QB), tm_o)
        num_tiles = B * K * (L // self.TM) * self.n_vt
        self.kernel(tm_s, tm_o, tma_a, mK_t, tma_q, mQ_t, tma_do, mDO_t, tLse.iterator, tKmin.iterator, tQmax.iterator,
                    tDV.iterator, a_layout, q_layout, do_layout, p_layout, L, nc, scale2).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_o: cute.TiledMma, tma_a: cute.CopyAtom, mK: cute.Tensor,
               tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_do: cute.CopyAtom, mDO: cute.Tensor, pLse: cute.Pointer,
               pKmin: cute.Pointer, pQmax: cute.Pointer, pDV: cute.Pointer, a_layout: cute.ComposedLayout,
               q_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout,
               L: Int32, nc: Int32, scale2: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, QB, C, VT = self.H, self.K, self.S, self.V, self.QB, self.C, self.VT
        key_tiles = L // self.TM
        vt = bidx % self.n_vt
        t0 = bidx // self.n_vt
        kt = t0 % key_tiles
        t1 = t0 // key_tiles
        kh = t1 % K
        b = t1 // K
        k0 = kt * self.TM
        ck = k0 // C
        # query blocks that can attend the tile: from its own rows (causal) to the last query whose segment
        # starts at or before the tile's last key, within the next chunk
        qmax = cute.make_tensor(pQmax + (b * L + k0 + self.TM - 1), cute.make_layout(1))[0]
        q_end = (ck + 2) * C - 1
        if q_end > L - 1:
            q_end = L - 1
        if qmax < q_end:
            q_end = qmax
        qb_lo = k0 // QB
        qb_hi = q_end // QB
        n_qb = qb_hi - qb_lo + 1
        n_blocks = n_qb * self.n_rep

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            p_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cw_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_cw)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        do_bytes = cute.size_in_bytes(BFloat16, cute.select(do_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=do_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=2, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cw)).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mbar.data_ptr(), num_stages=1, producer_group=cw_group, consumer_group=one).make_participants()
        # per-block column data of the compute warps (2 stages)
        sLse = smem.allocate_tensor(Float32, cute.make_layout(2 * QB), byte_alignment=16)
        sKmin = smem.allocate_tensor(Int32, cute.make_layout(2 * QB), byte_alignment=16)
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        sQ = smem.allocate_tensor(BFloat16, q_layout.outer, byte_alignment=1024, swizzle=q_layout.inner)
        sDO = smem.allocate_tensor(BFloat16, do_layout.outer, byte_alignment=1024, swizzle=do_layout.inner)

        gA = cute.local_tile(mK, (self.TM, S), (None, None, None, None))  # (128, S, L/128, 1, Kh, B)
        gQ = cute.local_tile(mQ, (QB, S), (None, None, None, None))  # (QB, S, L/QB, 1, H, B)
        gDO = cute.local_tile(mDO, (VT, QB), (None, None, None, None))  # (VT, QB, V/VT, L/QB, H, B)
        thr_s = tm_s.get_slice(0)
        thr_o = tm_o.get_slice(0)
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_s.partition_A(gA), 0, 3))
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_s.partition_B(gQ), 0, 3))
        tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(thr_o.partition_B(gDO), 0, 3))
        tCrA = tm_s.make_fragment_A(sA)
        tCrQ = tm_s.make_fragment_B(sQ)
        tCrDO = tm_o.make_fragment_B(sDO)
        s_fake = tm_s.make_fragment_C(cute.append(tm_s.partition_shape_C((self.TM, QB)), 2))
        o_fake = tm_o.make_fragment_C(tm_o.partition_shape_C((self.TM, VT)))
        tP_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), p_layout.outer)
        tPr_fake = tm_o.make_fragment_A(tP_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            ha = a_prod.acquire_and_advance()
            cute.copy(tma_a, tAg[(None, kt, 0, kh, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                hq = q_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, qb, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
                hd = do_prod.acquire_and_advance()
                cute.copy(tma_do, tDOg[(None, vt, qb, h, b)], tDOs[(None, hd.index)], tma_bar_ptr=hd.barrier)
            a_prod.tail()
            q_prod.tail()
            do_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            a_cons.wait()
            es = s_prod.acquire_and_advance()
            hq = q_cons.wait_and_advance()
            tS0 = tS[(None, None, None, es.index)]
            for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_s, tS0, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tS0)
            hq.release()
            es.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hq = q_cons.wait_and_advance()
                    tSj = tS[(None, None, None, es.index)]
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_s, tSj, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tSj)
                    hq.release()
                    es.commit()
                hp = p_cons.wait_and_advance()
                hd = do_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tPr, mode=[2])):
                    tm_o.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_o, tO, tPr[(None, None, kk)], tCrDO[(None, None, kk, hd.index)], tO)
                hd.release()
                hp.release()
            a_cons.release()
            s_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tO = cute.make_tensor(tmem_ptr + self.tmem_o, o_fake.layout)
            tPr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_p, dtype=BFloat16), tPr_fake.layout)
            trow = tidx % 128
            kpos = k0 + trow
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(QB), tcgen05.Pack.NONE), Float32)
            tS_r = [cute.flat_divide(tS[((None, None), 0, 0, i)], (self.TM, QB)) for i in range(2)]
            t2r_s = [tcgen05.make_tmem_copy(ld_s, tS_r[i][(None, None, 0, 0)]) for i in range(2)]
            tTR_s = [t2r_s[i].get_slice(trow).partition_S(tS_r[i]) for i in range(2)]
            tTR_s = [cute.group_modes(tTR_s[i], 3, cute.rank(tTR_s[i])) for i in range(2)]
            xs = cute.make_rmem_tensor(t2r_s[0].get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, QB))).shape, Float32)
            st_p = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(QB // 2), tcgen05.Unpack.NONE), BFloat16)
            r2t_p = tcgen05.make_tmem_copy(st_p, tPr)
            thr_p = r2t_p.get_slice(trow)
            tRT_tP = thr_p.partition_D(tPr)
            tRT_c = thr_p.partition_S(thr_o.partition_A(cute.make_identity_tensor((self.TM, QB))))
            pbf = cute.make_rmem_tensor(xs.shape, BFloat16)
            pbf_v = cute.make_tensor(pbf.iterator, cute.make_layout(tRT_c.shape))
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tO_r = cute.flat_divide(tO[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tO_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tO_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            cw_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.num_cw)

            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                q0 = qb * QB
                stg = j % 2
                # column data of this block: lse2[q], kmin[q] (stage j % 2; the barrier of the previous block
                # ordered its readers before these writes)
                if tidx < QB:
                    sLse[stg * QB + tidx] = cute.make_tensor(pLse + ((b * H + h) * L + q0 + tidx), cute.make_layout(1))[0]
                    sKmin[stg * QB + tidx] = cute.make_tensor(pKmin + (b * L + q0 + tidx), cute.make_layout(1))[0]
                cw_bar.arrive_and_wait()
                fs = s_cons.wait_and_advance()
                if fs.index == 0:
                    cute.copy(t2r_s[0], tTR_s[0][(None, None, None, 0)], xs)
                else:
                    cute.copy(t2r_s[1], tTR_s[1][(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                # mask: query q attends key k iff kmin[q] <= k <= q
                kmin_last = sKmin[stg * QB + QB - 1]
                if (q0 < k0 + self.TM - 1) | (kmin_last > k0):
                    for e in cutlass.range_constexpr(QB):
                        keep = (kpos >= sKmin[stg * QB + e]) & (kpos <= q0 + e)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                for e2 in cutlass.range_constexpr(QB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2),
                                                        (-sLse[stg * QB + 2 * e2], -sLse[stg * QB + 2 * e2 + 1]))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                pbf.store(xs.load().to(BFloat16))
                hp = p_prod.acquire_and_advance()
                cute.copy(r2t_p, pbf_v, tRT_tP)
                cute.arch.fence_view_async_tmem_store()
                hp.commit()
            hp = p_prod.acquire_and_advance()
            row_ptr = pDV + ((Int64(b * L + kpos) * K + kh) * V + vt * VT)
            for piece in cutlass.range_constexpr(VT // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store(ov.load().to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


class ScaBwdDk:
    """dk[keys of a 128-row tile] = scale sum_q dS^T Q for one (batch, kv head, key tile) per CTA.

    Per 64-query block (heads of the kv group x attending blocks, as ScaBwdDv): S^T = K Q^T and dP^T = V dO^T
    (tcgen05; K tile resident in shared memory, V tile resident in TMEM as the A operand), the compute warps
    (thread = key row) form P^T = exp2(S^T scale2 - lse2[q]) and dS^T = P^T (dP^T - delta[q]) scale into the
    shared-memory A operand of dK += dS^T Q (the Q block is reused as MN-major B). TMEM: S^T, dP^T, dK, V.

    For V > 256 (V and dO tiles too large for shared memory) dP^T is read instead from the fp32 GEMMs
    dPcur[(c, b, h)] = V_c dO_c^T and dPprev[(c - 1, b, h)] = V_{c-1} dO_c^T computed on the host."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, QB: int = 64, stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.QB = H, K, S, V, C, QB
        self.n_rep = H // K
        self.TM = 128
        self.load_dp = V > 256
        self.Vd = min(V, 256)  # width of the dO / V operand tiles (unused layouts in load_dp mode)
        self.stages = stages
        self.num_cw = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 64 == 0 and C % 128 == 0 and C % QB == 0
        self.tmem_s = 0
        self.tmem_dp = QB
        self.tmem_dk = 2 * QB
        self.tmem_v = 2 * QB + S
        total = self.tmem_v + (0 if self.load_dp else V // 2)
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        assert self.tmem_cols <= 512

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor,
                 tDelta: cute.Tensor, tKmin: cute.Tensor, tQmax: cute.Tensor, tDK: cute.Tensor, tDPprev: cute.Tensor,
                 tDPcur: cute.Tensor, B: Int32, L: Int32, scale2: Float32, scale: Float32, stream: cuda.CUstream):
        H, K, S, V, QB = self.H, self.K, self.S, self.Vd, self.QB
        nc = L // self.C
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mDO = cute.make_tensor(tDO.iterator, cute.make_layout((L, V, H, B), stride=(H * self.V, 1, self.V, L * H * self.V)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dp = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, QB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dk = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        a_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, QB, S), BFloat16, 1)
        q_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, QB, S), BFloat16, self.stages)
        qmn_layout = sm100_utils.make_smem_layout_b(tm_dk, (self.TM, S, QB), BFloat16, self.stages)  # same bytes
        do_layout = sm100_utils.make_smem_layout_b(tm_dp, (self.TM, QB, V), BFloat16, self.stages)
        ds_layout = sm100_utils.make_smem_layout_a(tm_dk, (self.TM, S, QB), BFloat16, 1)
        v_layout = sm100_utils.make_smem_layout_a(tm_dp, (self.TM, QB, V), BFloat16, 1)  # TMEM A operand
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQ, cute.slice_(q_layout, (None, None, None, 0)), (self.TM, QB, S), tm_s)
        tma_do, mDO_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mDO, cute.slice_(do_layout, (None, None, None, 0)), (self.TM, QB, V), tm_dp)
        num_tiles = B * K * (L // self.TM)
        self.kernel(tm_s, tm_dp, tm_dk, tma_a, mK_t, tma_q, mQ_t, tma_do, mDO_t, tV.iterator, tLse.iterator, tDelta.iterator,
                    tKmin.iterator, tQmax.iterator, tDK.iterator, tDPprev.iterator, tDPcur.iterator, a_layout, q_layout,
                    qmn_layout, do_layout, ds_layout, v_layout, B, L, nc, scale2, scale).launch(
            grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_dp: cute.TiledMma, tm_dk: cute.TiledMma, tma_a: cute.CopyAtom, mK: cute.Tensor,
               tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_do: cute.CopyAtom, mDO: cute.Tensor, pV: cute.Pointer,
               pLse: cute.Pointer, pDelta: cute.Pointer, pKmin: cute.Pointer, pQmax: cute.Pointer, pDK: cute.Pointer,
               pDPprev: cute.Pointer, pDPcur: cute.Pointer, a_layout: cute.ComposedLayout, q_layout: cute.ComposedLayout,
               qmn_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout, ds_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, B: Int32, L: Int32, nc: Int32, scale2: Float32, scale: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, QB, C = self.H, self.K, self.S, self.V, self.QB, self.C
        key_tiles = L // self.TM
        kt = bidx % key_tiles
        t1 = bidx // key_tiles
        kh = t1 % K
        b = t1 // K
        k0 = kt * self.TM
        ck = k0 // C
        qmax = cute.make_tensor(pQmax + (b * L + k0 + self.TM - 1), cute.make_layout(1))[0]
        q_end = (ck + 2) * C - 1
        if q_end > L - 1:
            q_end = L - 1
        if qmax < q_end:
            q_end = qmax
        qb_lo = k0 // QB
        qb_hi = q_end // QB
        n_qb = qb_hi - qb_lo + 1
        n_blocks = n_qb * self.n_rep

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_q)
            if const_expr(not self.load_dp):
                cpasync.prefetch_descriptor(tma_do)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            do_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            dp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            ds_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        v_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cw_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_cw)
        cw_warps = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cw)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
        do_bytes = cute.size_in_bytes(BFloat16, cute.select(do_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        q_prod, q_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=q_bytes).make_participants()
        do_prod, do_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.do_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=do_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        dp_prod, dp_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dp_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        ds_prod, ds_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.ds_mbar.data_ptr(), num_stages=1, producer_group=cw_group, consumer_group=one).make_participants()
        sLse = smem.allocate_tensor(Float32, cute.make_layout(2 * QB), byte_alignment=16)
        sDelta = smem.allocate_tensor(Float32, cute.make_layout(2 * QB), byte_alignment=16)
        sKmin = smem.allocate_tensor(Int32, cute.make_layout(2 * QB), byte_alignment=16)
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        pQ = smem.allocate_array(BFloat16, cute.cosize(q_layout.outer), byte_alignment=1024)
        sQ = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=q_layout.inner), q_layout.outer)
        sQmn = cute.make_tensor(cute.recast_ptr(pQ, swizzle_=qmn_layout.inner), qmn_layout.outer)
        if const_expr(not self.load_dp):
            sDO = smem.allocate_tensor(BFloat16, do_layout.outer, byte_alignment=1024, swizzle=do_layout.inner)
        pDSs = smem.allocate_array(BFloat16, cute.cosize(ds_layout.outer), byte_alignment=1024)
        sDS = cute.make_tensor(cute.recast_ptr(pDSs, swizzle_=ds_layout.inner), ds_layout.outer)

        gA = cute.local_tile(mK, (self.TM, S), (None, None, None, None))
        gQ = cute.local_tile(mQ, (QB, S), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_dp = tm_dp.get_slice(0)
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_s.partition_A(gA), 0, 3))
        tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_s.partition_B(gQ), 0, 3))
        if const_expr(not self.load_dp):
            gDO = cute.local_tile(mDO, (QB, self.Vd), (None, None, None, None))
            tDOs, tDOg = cpasync.tma_partition(tma_do, 0, cute.make_layout(1), cute.group_modes(sDO, 0, 3), cute.group_modes(thr_dp.partition_B(gDO), 0, 3))
            tCrDO = tm_dp.make_fragment_B(sDO)
        tCrA = tm_s.make_fragment_A(sA)
        tCrQ = tm_s.make_fragment_B(sQ)
        tCrQmn = tm_dk.make_fragment_B(sQmn)
        tCrDS = tm_dk.make_fragment_A(sDS)
        s_fake = tm_s.make_fragment_C(tm_s.partition_shape_C((self.TM, QB)))
        dk_fake = tm_dk.make_fragment_C(tm_dk.partition_shape_C((self.TM, S)))
        tV_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), v_layout.outer)
        tVr_fake = tm_dp.make_fragment_A(tV_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            ha = a_prod.acquire_and_advance()
            cute.copy(tma_a, tAg[(None, kt, 0, kh, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                hq = q_prod.acquire_and_advance()
                cute.copy(tma_q, tQg[(None, qb, 0, h, b)], tQs[(None, hq.index)], tma_bar_ptr=hq.barrier)
                if const_expr(not self.load_dp):
                    hd = do_prod.acquire_and_advance()
                    cute.copy(tma_do, tDOg[(None, qb, 0, h, b)], tDOs[(None, hd.index)], tma_bar_ptr=hd.barrier)
            a_prod.tail()
            q_prod.tail()
            if const_expr(not self.load_dp):
                do_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            if const_expr(not self.load_dp):
                v_bar.arrive_and_wait()  # V tile stored in TMEM by the compute warps
            a_cons.wait()
            # the Q stage of a block is held until its dK MMA: q_cons waits (one block ahead), q_rel releases
            q_rel = q_cons.clone()
            # S^T_0 and dP^T_0
            es = s_prod.acquire_and_advance()
            hq = q_cons.wait_and_advance()
            for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tS)
            es.commit()
            if const_expr(not self.load_dp):
                ed = dp_prod.acquire_and_advance()
                hd = do_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tVr, mode=[2])):
                    tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                    cute.gemm(tm_dp, tDP, tVr[(None, None, kk)], tCrDO[(None, None, kk, hd.index)], tDP)
                hd.release()
                ed.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hq = q_cons.wait_and_advance()
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrQ[(None, None, kk, hq.index)], tS)
                    es.commit()
                    if const_expr(not self.load_dp):
                        ed = dp_prod.acquire_and_advance()
                        hd = do_cons.wait_and_advance()
                        for kk in cutlass.range_constexpr(cute.size(tVr, mode=[2])):
                            tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                            cute.gemm(tm_dp, tDP, tVr[(None, None, kk)], tCrDO[(None, None, kk, hd.index)], tDP)
                        hd.release()
                        ed.commit()
                # dK += dS^T_j Q_j
                hs = ds_cons.wait_and_advance()
                qj = q_rel.current_handle().index
                for kk in cutlass.range_constexpr(cute.size(tCrDS, mode=[2])):
                    tm_dk.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_dk, tDK, tCrDS[(None, None, kk, 0)], tCrQmn[(None, None, kk, qj)], tDK)
                q_rel.release()
                q_rel.advance()
                hs.release()
            a_cons.release()
            s_prod.tail()
            if const_expr(not self.load_dp):
                dp_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDK = cute.make_tensor(tmem_ptr + self.tmem_dk, dk_fake.layout)
            tVr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_v, dtype=BFloat16), tVr_fake.layout)
            trow = tidx % 128
            kpos = k0 + trow
            if const_expr(not self.load_dp):
                # V tile -> TMEM (A operand of dP^T): this thread's key row
                st_v = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(V // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_v = tcgen05.make_tmem_copy(st_v, tVr)
                thr_v = r2t_v.get_slice(trow)
                tRT_tV = thr_v.partition_D(tVr)
                tRT_cv = thr_v.partition_S(thr_dp.partition_A(cute.make_identity_tensor((self.TM, V))))
                rV = cute.make_rmem_tensor((V,), BFloat16)
                v_ptr = pV + (Int64(b * L + kpos) * K + kh) * V
                for e8 in cutlass.range_constexpr(V // 8):
                    cute.make_tensor(rV.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((v_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_v, cute.make_tensor(rV.iterator, cute.make_layout(tRT_cv.shape)), tRT_tV)
                cute.arch.fence_view_async_tmem_store()
                v_bar.arrive_and_wait()
            # dP^T matrices of this key tile's chunk (load_dp mode): rows = keys of chunk ck, columns = queries
            m_idx = Int64((ck * B + b) * H) * C * C  # + h * C * C per head
            dp_row = Int64(kpos - ck * C) * C
            # S^T / dP^T loads (QB columns of this thread's row)
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(QB), tcgen05.Pack.NONE), Float32)
            tS_r = cute.flat_divide(tS[((None, None), 0, 0)], (self.TM, QB))
            tDP_r = cute.flat_divide(tDP[((None, None), 0, 0)], (self.TM, QB))
            t2r_s = tcgen05.make_tmem_copy(ld_s, tS_r[(None, None, 0, 0)])
            t2r_dp = tcgen05.make_tmem_copy(ld_s, tDP_r[(None, None, 0, 0)])
            tTR_s = t2r_s.get_slice(trow).partition_S(tS_r)
            tTR_s = cute.group_modes(tTR_s, 3, cute.rank(tTR_s))
            tTR_dp = t2r_dp.get_slice(trow).partition_S(tDP_r)
            tTR_dp = cute.group_modes(tTR_dp, 3, cute.rank(tTR_dp))
            xs = cute.make_rmem_tensor(t2r_s.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, QB))).shape, Float32)
            xd = cute.make_rmem_tensor(xs.shape, Float32)
            dsb = cute.make_rmem_tensor(xs.shape, BFloat16)
            # dK pieces (epilogue)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tDK_r = cute.flat_divide(tDK[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tDK_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tDK_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            cw_bar = pipeline.NamedBarrier(barrier_id=3, num_threads=32 * self.num_cw)
            # column data of block 0
            nxt_lse = Float32(0.0)
            nxt_delta = Float32(0.0)
            nxt_kmin = Int32(0)
            if tidx < QB:
                h0 = kh * self.n_rep
                nxt_lse = cute.make_tensor(pLse + ((b * H + h0) * L + qb_lo * QB + tidx), cute.make_layout(1))[0]
                nxt_delta = cute.make_tensor(pDelta + ((b * H + h0) * L + qb_lo * QB + tidx), cute.make_layout(1))[0]
                nxt_kmin = cute.make_tensor(pKmin + (b * L + qb_lo * QB + tidx), cute.make_layout(1))[0]
            for j in cutlass.range(n_blocks):
                r = j // n_qb
                qb = qb_lo + j - r * n_qb
                h = kh * self.n_rep + r
                q0 = qb * QB
                stg = j % 2
                if tidx < QB:
                    sLse[stg * QB + tidx] = nxt_lse
                    sDelta[stg * QB + tidx] = nxt_delta
                    sKmin[stg * QB + tidx] = nxt_kmin
                cw_bar.arrive_and_wait()
                # prefetch the next block's column data (latency hidden behind the S^T wait)
                if j + 1 < n_blocks:
                    if tidx < QB:
                        r1 = (j + 1) // n_qb
                        q1 = (qb_lo + (j + 1) - r1 * n_qb) * QB
                        h1 = kh * self.n_rep + r1
                        nxt_lse = cute.make_tensor(pLse + ((b * H + h1) * L + q1 + tidx), cute.make_layout(1))[0]
                        nxt_delta = cute.make_tensor(pDelta + ((b * H + h1) * L + q1 + tidx), cute.make_layout(1))[0]
                        nxt_kmin = cute.make_tensor(pKmin + (b * L + q1 + tidx), cute.make_layout(1))[0]
                # P^T
                fs = s_cons.wait_and_advance()
                cute.copy(t2r_s, tTR_s[(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                kmin_last = sKmin[stg * QB + QB - 1]
                if (q0 < k0 + self.TM - 1) | (kmin_last > k0):
                    for e in cutlass.range_constexpr(QB):
                        keep = (kpos >= sKmin[stg * QB + e]) & (kpos <= q0 + e)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                for e2 in cutlass.range_constexpr(QB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2),
                                                        (-sLse[stg * QB + 2 * e2], -sLse[stg * QB + 2 * e2 + 1]))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                # dS^T = P^T (dP^T - delta) scale
                if const_expr(self.load_dp):
                    cq = q0 // C
                    dp_base = pDPcur + (m_idx + Int64(h) * C * C + dp_row + Int64(q0 - cq * C))
                    if cq != ck:
                        dp_base = pDPprev + (m_idx + Int64(h) * C * C + dp_row + Int64(q0 - cq * C))
                    for e4 in cutlass.range_constexpr(QB // 4):
                        cute.make_tensor(xd.iterator + e4 * 4, cute.make_layout(4)).store(
                            cute.make_tensor((dp_base + e4 * 4).align(16), cute.make_layout(4)).load())
                else:
                    fd = dp_cons.wait_and_advance()
                    cute.copy(t2r_dp, tTR_dp[(None, None, None, 0)], xd)
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        fd.release()
                for e in cutlass.range_constexpr(QB):
                    xd[e] = xs[e] * (xd[e] - sDelta[stg * QB + e]) * scale
                dsb.store(xd.load().to(BFloat16))
                # shared-memory A operand of dK (K-major, 128 x QB, 128-byte swizzle), free once dK_{j-1} is done
                hs = ds_prod.acquire_and_advance()
                for e8 in cutlass.range_constexpr(QB // 8):
                    cute.make_tensor((pDSs + (64 * trow + 8 * (e8 ^ (trow % 8)))).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
                cute.arch.fence_view_async_shared()
                hs.commit()
            # dK epilogue (the last dK MMA completed when the dS buffer is released again)
            hs = ds_prod.acquire_and_advance()
            row_ptr = pDK + (Int64(b * L + kpos) * K + kh) * S
            for piece in cutlass.range_constexpr(S // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store(ov.load().to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


def sca_bwd(q, k, v, y, lse2, dout, chunk_size, scale, kmin, qmax):
    """Gradients of sca_fwd: dq [B, L, H, S], dk [B, L, K, S], dv [B, L, K, V] (bf16)."""
    B, L, H, S = q.shape
    K, V = v.shape[2], v.shape[3]
    C = chunk_size
    nc = L // C
    dev = q.device
    dout = dout.contiguous()
    # delta[b, h, q] = sum_v dO O
    delta = torch.empty(B, H, L, dtype=torch.float32, device=dev)
    f_delta = _compiled(("sca_rowdot", H, V), lambda: RowDot(H, V), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), Int32(0), Int32(0)))
    f_delta(dout.reshape(-1), y.reshape(-1), delta.reshape(-1), B, L)
    dv = torch.empty(B, L, K, V, dtype=torch.bfloat16, device=dev)
    f_dv = _compiled(("sca_bwd_dv", H, K, S, V, C), lambda: ScaBwdDv(H, K, S, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake(Int32, 4),
        _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    f_dv(q.reshape(-1), k.reshape(-1), dout.reshape(-1), lse2.reshape(-1), kmin.reshape(-1), qmax.reshape(-1), dv.reshape(-1),
         B, L, float(scale * LOG2E))
    if V > 256:
        # dP for the dK / dQ kernels (fp32 GEMMs): dPcur[(c, b, h)] = V_c dO_c^T, dPprev[(c - 1, b, h)] = V_{c-1} dO_c^T
        f_cp = _compiled(("sca_cmcopy", H, K, V, C), lambda: ChunkMajorCopy(H, K, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
        f_cp2 = _compiled(("sca_cmcopy", H, H, V, C), lambda: ChunkMajorCopy(H, H, V, C), lambda: (
            _fake(BFloat16), _fake(BFloat16), Int32(0), Int32(0)))
        vc = torch.empty(nc, B, H, C, V, dtype=torch.bfloat16, device=dev)  # heads repeated
        doc = torch.empty(nc, B, H, C, V, dtype=torch.bfloat16, device=dev)
        f_cp(v.reshape(-1), vc.reshape(-1), B, L)
        f_cp2(dout.reshape(-1), doc.reshape(-1), B, L)
        dp_cur = torch.bmm(vc.view(-1, C, V), doc.view(-1, C, V).transpose(1, 2), out_dtype=torch.float32)
        if nc > 1:
            dp_prev = torch.bmm(vc[:-1].reshape(-1, C, V), doc[1:].reshape(-1, C, V).transpose(1, 2), out_dtype=torch.float32)
        else:
            dp_prev = dp_cur
    else:
        dp_cur = dp_prev = torch.zeros(1, dtype=torch.float32, device=dev)
    dk = torch.empty(B, L, K, S, dtype=torch.bfloat16, device=dev)
    f_dk = _compiled(("sca_bwd_dk", H, K, S, V, C), lambda: ScaBwdDk(H, K, S, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
        _fake(Int32, 4), _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0),
        Float32(0.0), Float32(0.0)))
    f_dk(q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1), lse2.reshape(-1), delta.reshape(-1), kmin.reshape(-1),
         qmax.reshape(-1), dk.reshape(-1), _col2d(dp_prev), _col2d(dp_cur), B, L, float(scale * LOG2E), float(scale))
    dq = torch.empty(B, L, H, S, dtype=torch.bfloat16, device=dev)
    f_dq = _compiled(("sca_bwd_dq", H, K, S, V, C), lambda: ScaBwdDq(H, K, S, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Float32, 4),
        _fake(Int32, 4), _fake(BFloat16), _fake2(Float32), _fake2(Float32), Int32(0), Int32(0), Float32(0.0), Float32(0.0)))
    f_dq(q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1), lse2.reshape(-1), delta.reshape(-1), kmin.reshape(-1),
         dq.reshape(-1), _col2d(dp_prev), _col2d(dp_cur), B, L, float(scale * LOG2E), float(scale))
    return dq, dk, dv


class ScaBwdDq:
    """dq[queries of a 128-row tile] = scale sum_k dS K for one (batch, head, chunk, query tile) per CTA.

    Mirror of ScaBwdDk with queries as rows: the q tile is resident in shared memory (A of S = Q K^T), the dO tile
    in TMEM (A of dP = dO V^T), 64-key blocks of K (K-major B of S, MN-major B of dQ += dS K over the same bytes)
    and V (K-major B of dP) stream through 2-stage rings; the compute warps (thread = query row) form
    P = exp2(S scale2 - lse2) and dS = P (dP - delta) scale into the shared-memory A operand of the dQ MMA.
    TMEM: S, dP, dQ, dO. For V > 256 dP is read from the host-side fp32 GEMMs (see ScaBwdDk)."""

    def __init__(self, H: int, K: int, S: int, V: int, C: int, KB: int = 64, stages: int = 2):
        self.H, self.K, self.S, self.V, self.C, self.KB = H, K, S, V, C, KB
        self.n_rep = H // K
        self.TM = 128
        self.load_dp = V > 256
        self.Vd = min(V, 256)
        self.stages = stages
        self.num_cw = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        self.row_tiles = C // self.TM
        assert H % K == 0 and S % 64 == 0 and S <= 256 and V % 64 == 0 and C % 128 == 0 and C % KB == 0
        self.tmem_s = 0
        self.tmem_dp = KB
        self.tmem_dq = 2 * KB
        self.tmem_do = 2 * KB + S
        total = self.tmem_do + (0 if self.load_dp else V // 2)
        self.tmem_cols = max(32, 1 << (total - 1).bit_length())
        assert self.tmem_cols <= 512

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tDO: cute.Tensor, tLse: cute.Tensor,
                 tDelta: cute.Tensor, tKmin: cute.Tensor, tDQ: cute.Tensor, tDPprev: cute.Tensor, tDPcur: cute.Tensor,
                 B: Int32, L: Int32, scale2: Float32, scale: Float32, stream: cuda.CUstream):
        H, K, S, V, KB = self.H, self.K, self.S, self.Vd, self.KB
        nc = L // self.C
        mQ = cute.make_tensor(tQ.iterator, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        mK = cute.make_tensor(tK.iterator, cute.make_layout((L, S, K, B), stride=(K * S, 1, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((L, V, K, B), stride=(K * self.V, 1, self.V, L * K * self.V)))
        tm_s = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dp = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, KB, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        tm_dq = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.TM, S, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        a_layout = sm100_utils.make_smem_layout_a(tm_s, (self.TM, KB, S), BFloat16, 1)
        k_layout = sm100_utils.make_smem_layout_b(tm_s, (self.TM, KB, S), BFloat16, self.stages)
        kmn_layout = sm100_utils.make_smem_layout_b(tm_dq, (self.TM, S, KB), BFloat16, self.stages)
        v_layout = sm100_utils.make_smem_layout_b(tm_dp, (self.TM, KB, V), BFloat16, self.stages)
        ds_layout = sm100_utils.make_smem_layout_a(tm_dq, (self.TM, S, KB), BFloat16, 1)
        do_layout = sm100_utils.make_smem_layout_a(tm_dp, (self.TM, KB, V), BFloat16, 1)  # TMEM A operand
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mQ, cute.slice_(a_layout, (None, None, None, 0)), (self.TM, KB, S), tm_s)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mK, cute.slice_(k_layout, (None, None, None, 0)), (self.TM, KB, S), tm_s)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, cute.slice_(v_layout, (None, None, None, 0)), (self.TM, KB, V), tm_dp)
        num_tiles = B * nc * H * self.row_tiles
        self.kernel(tm_s, tm_dp, tm_dq, tma_a, mQ_t, tma_k, mK_t, tma_v, mV_t, tDO.iterator, tLse.iterator, tDelta.iterator,
                    tKmin.iterator, tDQ.iterator, tDPprev.iterator, tDPcur.iterator, a_layout, k_layout, kmn_layout, v_layout,
                    ds_layout, do_layout, B, L, nc, scale2, scale).launch(grid=[num_tiles, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tm_s: cute.TiledMma, tm_dp: cute.TiledMma, tm_dq: cute.TiledMma, tma_a: cute.CopyAtom, mQ: cute.Tensor,
               tma_k: cute.CopyAtom, mK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor, pDO: cute.Pointer,
               pLse: cute.Pointer, pDelta: cute.Pointer, pKmin: cute.Pointer, pDQ: cute.Pointer, pDPprev: cute.Pointer,
               pDPcur: cute.Pointer, a_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout, kmn_layout: cute.ComposedLayout,
               v_layout: cute.ComposedLayout, ds_layout: cute.ComposedLayout, do_layout: cute.ComposedLayout,
               B: Int32, L: Int32, nc: Int32, scale2: Float32, scale: Float32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, S, V, KB, C = self.H, self.K, self.S, self.V, self.KB, self.C
        rt = bidx % self.row_tiles
        t1 = bidx // self.row_tiles
        h = t1 % H
        t2 = t1 // H
        c = t2 % nc
        b = t2 // nc
        kh = h // self.n_rep
        q0 = c * C + rt * self.TM
        kstart = Int32(0)
        if c > 0:
            kstart = (c - 1) * C
        kmin0 = cute.make_tensor(pKmin + (b * L + q0), cute.make_layout(1))[0]
        kb_lo = (kmin0 - kstart) // KB
        if kb_lo < 0:
            kb_lo = Int32(0)
        kb_hi = (q0 + self.TM - 1 - kstart) // KB
        n_blocks = kb_hi - kb_lo + 1
        kblk0 = kstart // KB + kb_lo

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_k)
            if const_expr(not self.load_dp):
                cpasync.prefetch_descriptor(tma_v)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            s_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            dp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            ds_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_cw + 1))
        do_bar = pipeline.NamedBarrier(barrier_id=4, num_threads=32 * (self.num_cw + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cw_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_cw)
        cw_warps = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_cw)
        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))
        v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2]))
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=one,
            tx_count=a_bytes).make_participants()
        k_prod, k_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=k_bytes).make_participants()
        v_prod, v_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.v_mbar.data_ptr(), num_stages=self.stages, producer_group=one, consumer_group=one,
            tx_count=v_bytes).make_participants()
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.s_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        dp_prod, dp_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.dp_mbar.data_ptr(), num_stages=1, producer_group=one, consumer_group=cw_warps).make_participants()
        ds_prod, ds_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.ds_mbar.data_ptr(), num_stages=1, producer_group=cw_group, consumer_group=one).make_participants()
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
        pK = smem.allocate_array(BFloat16, cute.cosize(k_layout.outer), byte_alignment=1024)
        sK = cute.make_tensor(cute.recast_ptr(pK, swizzle_=k_layout.inner), k_layout.outer)
        sKmn = cute.make_tensor(cute.recast_ptr(pK, swizzle_=kmn_layout.inner), kmn_layout.outer)
        if const_expr(not self.load_dp):
            sV = smem.allocate_tensor(BFloat16, v_layout.outer, byte_alignment=1024, swizzle=v_layout.inner)
        pDSs = smem.allocate_array(BFloat16, cute.cosize(ds_layout.outer), byte_alignment=1024)
        sDS = cute.make_tensor(cute.recast_ptr(pDSs, swizzle_=ds_layout.inner), ds_layout.outer)

        gA = cute.local_tile(mQ, (self.TM, S), (None, None, None, None))
        gK = cute.local_tile(mK, (KB, S), (None, None, None, None))
        thr_s = tm_s.get_slice(0)
        thr_dp = tm_dp.get_slice(0)
        tAs, tAg = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3), cute.group_modes(thr_s.partition_A(gA), 0, 3))
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3), cute.group_modes(thr_s.partition_B(gK), 0, 3))
        if const_expr(not self.load_dp):
            gV = cute.local_tile(mV, (KB, self.Vd), (None, None, None, None))
            tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3), cute.group_modes(thr_dp.partition_B(gV), 0, 3))
            tCrV = tm_dp.make_fragment_B(sV)
        tCrA = tm_s.make_fragment_A(sA)
        tCrK = tm_s.make_fragment_B(sK)
        tCrKmn = tm_dq.make_fragment_B(sKmn)
        tCrDS = tm_dq.make_fragment_A(sDS)
        s_fake = tm_s.make_fragment_C(tm_s.partition_shape_C((self.TM, KB)))
        dq_fake = tm_dq.make_fragment_C(tm_dq.partition_shape_C((self.TM, S)))
        tDO_fake = cute.make_tensor(cute.make_ptr(BFloat16, 0, cute.AddressSpace.tmem), do_layout.outer)
        tDOr_fake = tm_dp.make_fragment_A(tDO_fake)[(None, None, None, 0)]

        if warp == self.tma_warp:
            ha = a_prod.acquire_and_advance()
            cute.copy(tma_a, tAg[(None, q0 // self.TM, 0, h, b)], tAs[(None, ha.index)], tma_bar_ptr=ha.barrier)
            for j in cutlass.range(n_blocks):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, kblk0 + j, 0, kh, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                if const_expr(not self.load_dp):
                    hv = v_prod.acquire_and_advance()
                    cute.copy(tma_v, tVg[(None, kblk0 + j, 0, kh, b)], tVs[(None, hv.index)], tma_bar_ptr=hv.barrier)
            a_prod.tail()
            k_prod.tail()
            if const_expr(not self.load_dp):
                v_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDQ = cute.make_tensor(tmem_ptr + self.tmem_dq, dq_fake.layout)
            tDOr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_do, dtype=BFloat16), tDOr_fake.layout)
            if const_expr(not self.load_dp):
                do_bar.arrive_and_wait()  # dO tile stored in TMEM by the compute warps
            a_cons.wait()
            k_rel = k_cons.clone()
            es = s_prod.acquire_and_advance()
            hk = k_cons.wait_and_advance()
            for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tS)
            es.commit()
            if const_expr(not self.load_dp):
                ed = dp_prod.acquire_and_advance()
                hv = v_cons.wait_and_advance()
                for kk in cutlass.range_constexpr(cute.size(tDOr, mode=[2])):
                    tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                    cute.gemm(tm_dp, tDP, tDOr[(None, None, kk)], tCrV[(None, None, kk, hv.index)], tDP)
                hv.release()
                ed.commit()
            for j in cutlass.range(n_blocks):
                if j + 1 < n_blocks:
                    es = s_prod.acquire_and_advance()
                    hk = k_cons.wait_and_advance()
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        tm_s.set(tcgen05.Field.ACCUMULATE, kk > 0)
                        cute.gemm(tm_s, tS, tCrA[(None, None, kk, 0)], tCrK[(None, None, kk, hk.index)], tS)
                    es.commit()
                    if const_expr(not self.load_dp):
                        ed = dp_prod.acquire_and_advance()
                        hv = v_cons.wait_and_advance()
                        for kk in cutlass.range_constexpr(cute.size(tDOr, mode=[2])):
                            tm_dp.set(tcgen05.Field.ACCUMULATE, kk > 0)
                            cute.gemm(tm_dp, tDP, tDOr[(None, None, kk)], tCrV[(None, None, kk, hv.index)], tDP)
                        hv.release()
                        ed.commit()
                hs = ds_cons.wait_and_advance()
                kj = k_rel.current_handle().index
                for kk in cutlass.range_constexpr(cute.size(tCrDS, mode=[2])):
                    tm_dq.set(tcgen05.Field.ACCUMULATE, (j > 0) | (kk > 0))
                    cute.gemm(tm_dq, tDQ, tCrDS[(None, None, kk, 0)], tCrKmn[(None, None, kk, kj)], tDQ)
                k_rel.release()
                k_rel.advance()
                hs.release()
            a_cons.release()
            s_prod.tail()
            if const_expr(not self.load_dp):
                dp_prod.tail()
        else:
            tmem.allocate(self.tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tS = cute.make_tensor(tmem_ptr + self.tmem_s, s_fake.layout)
            tDP = cute.make_tensor(tmem_ptr + self.tmem_dp, s_fake.layout)
            tDQ = cute.make_tensor(tmem_ptr + self.tmem_dq, dq_fake.layout)
            tDOr = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.tmem_do, dtype=BFloat16), tDOr_fake.layout)
            trow = tidx % 128
            qpos = q0 + trow
            kmin_row = cute.make_tensor(pKmin + (b * L + qpos), cute.make_layout(1))[0]
            kmin_last = cute.make_tensor(pKmin + (b * L + q0 + self.TM - 1), cute.make_layout(1))[0]
            lse_row = cute.make_tensor(pLse + ((b * H + h) * L + qpos), cute.make_layout(1))[0]
            delta_row = cute.make_tensor(pDelta + ((b * H + h) * L + qpos), cute.make_layout(1))[0]
            if const_expr(not self.load_dp):
                # dO tile -> TMEM (A operand of dP): this thread's query row
                st_do = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(V // 2), tcgen05.Unpack.NONE), BFloat16)
                r2t_do = tcgen05.make_tmem_copy(st_do, tDOr)
                thr_do = r2t_do.get_slice(trow)
                tRT_tDO = thr_do.partition_D(tDOr)
                tRT_cdo = thr_do.partition_S(thr_dp.partition_A(cute.make_identity_tensor((self.TM, V))))
                rDO = cute.make_rmem_tensor((V,), BFloat16)
                do_ptr = pDO + (Int64(b * L + qpos) * H + h) * V
                for e8 in cutlass.range_constexpr(V // 8):
                    cute.make_tensor(rDO.iterator + e8 * 8, cute.make_layout(8)).store(
                        cute.make_tensor((do_ptr + e8 * 8).align(16), cute.make_layout(8)).load())
                cute.copy(r2t_do, cute.make_tensor(rDO.iterator, cute.make_layout(tRT_cdo.shape)), tRT_tDO)
                cute.arch.fence_view_async_tmem_store()
                do_bar.arrive_and_wait()
            # dP matrices (load_dp mode): dPcur[(c, b, h)][key - cC][q - cC], dPprev[(c - 1, b, h)][key - (c-1)C][q - cC]
            q_in = Int64(qpos - c * C)
            cur_base = pDPcur + (Int64((c * B + b) * H + h) * C * C + q_in)
            prev_base = pDPprev + (Int64(((c - 1) * B + b) * H + h) * C * C + q_in)
            ld_s = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(KB), tcgen05.Pack.NONE), Float32)
            tS_r = cute.flat_divide(tS[((None, None), 0, 0)], (self.TM, KB))
            tDP_r = cute.flat_divide(tDP[((None, None), 0, 0)], (self.TM, KB))
            t2r_s = tcgen05.make_tmem_copy(ld_s, tS_r[(None, None, 0, 0)])
            t2r_dp = tcgen05.make_tmem_copy(ld_s, tDP_r[(None, None, 0, 0)])
            tTR_s = t2r_s.get_slice(trow).partition_S(tS_r)
            tTR_s = cute.group_modes(tTR_s, 3, cute.rank(tTR_s))
            tTR_dp = t2r_dp.get_slice(trow).partition_S(tDP_r)
            tTR_dp = cute.group_modes(tTR_dp, 3, cute.rank(tTR_dp))
            xs = cute.make_rmem_tensor(t2r_s.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, KB))).shape, Float32)
            xd = cute.make_rmem_tensor(xs.shape, Float32)
            dsb = cute.make_rmem_tensor(xs.shape, BFloat16)
            ld_o = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32), tcgen05.Pack.NONE), Float32)
            tDQ_r = cute.flat_divide(tDQ[((None, None), 0, 0)], (self.TM, 32))
            t2r_o = tcgen05.make_tmem_copy(ld_o, tDQ_r[(None, None, 0, 0)])
            tTR_o = t2r_o.get_slice(trow).partition_S(tDQ_r)
            tTR_o = cute.group_modes(tTR_o, 3, cute.rank(tTR_o))
            ov = cute.make_rmem_tensor(t2r_o.get_slice(trow).partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
            obf = cute.make_rmem_tensor(ov.shape, BFloat16)
            neg_lse = -lse_row
            for j in cutlass.range(n_blocks):
                bstart = kstart + (kb_lo + j) * KB
                fs = s_cons.wait_and_advance()
                cute.copy(t2r_s, tTR_s[(None, None, None, 0)], xs)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    fs.release()
                if (bstart < kmin_last) | (bstart + KB - 1 > q0):
                    lo = kmin_row - bstart
                    hi = qpos - bstart
                    for e in cutlass.range_constexpr(KB):
                        keep = (e >= lo) & (e <= hi)
                        xs[e] = cutlass.select_(keep, xs[e], Float32(float('-inf')))
                for e2 in cutlass.range_constexpr(KB // 2):
                    y0, y1 = cute.arch.fma_packed_f32x2((xs[2 * e2], xs[2 * e2 + 1]), (scale2, scale2), (neg_lse, neg_lse))
                    xs[2 * e2] = cute.math.exp2(y0, fastmath=True)
                    xs[2 * e2 + 1] = cute.math.exp2(y1, fastmath=True)
                if const_expr(self.load_dp):
                    ckb = bstart // C
                    dp_base = cur_base + Int64(bstart - ckb * C) * C
                    if ckb != c:
                        dp_base = prev_base + Int64(bstart - ckb * C) * C
                    for e in cutlass.range_constexpr(KB):
                        xd[e] = cute.make_tensor(dp_base + e * C, cute.make_layout(1))[0]
                else:
                    fd = dp_cons.wait_and_advance()
                    cute.copy(t2r_dp, tTR_dp[(None, None, None, 0)], xd)
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        fd.release()
                for e in cutlass.range_constexpr(KB):
                    xd[e] = xs[e] * (xd[e] - delta_row) * scale
                dsb.store(xd.load().to(BFloat16))
                hs = ds_prod.acquire_and_advance()
                for e8 in cutlass.range_constexpr(KB // 8):
                    cute.make_tensor((pDSs + (64 * trow + 8 * (e8 ^ (trow % 8)))).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(dsb.iterator + e8 * 8, cute.make_layout(8)).load())
                cute.arch.fence_view_async_shared()
                hs.commit()
            hs = ds_prod.acquire_and_advance()
            row_ptr = pDQ + (Int64(b * L + qpos) * H + h) * S
            for piece in cutlass.range_constexpr(S // 32):
                cute.copy(t2r_o, tTR_o[(None, None, None, piece)], ov)
                cute.arch.fence_view_async_tmem_load()
                obf.store(ov.load().to(BFloat16))
                for e8 in cutlass.range_constexpr(4):
                    cute.make_tensor((row_ptr + (piece * 32 + e8 * 8)).align(16), cute.make_layout(8)).store(
                        cute.make_tensor(obf.iterator + e8 * 8, cute.make_layout(8)).load())
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_cw)
            tmem.free(tmem_ptr)


class ChunkMajorCopy:
    """out[c, b, h, t, :] = x[b, c C + t, h // n_rep, :] (bf16 rows of V): the chunk-major, head-repeated copies that
    give the dP GEMMs of the large-V backward uniform batch strides (torch's permute copies run at ~1.7 TB/s)."""

    def __init__(self, H: int, K: int, V: int, C: int):
        self.H, self.K, self.V, self.C = H, K, V, C
        self.n_rep = H // K
        assert V % 8 == 0 and H % K == 0

    @cute.jit
    def __call__(self, tX: cute.Tensor, tOut: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        rows = B * L * self.H
        self.kernel(tX.iterator, tOut.iterator, B, L).launch(grid=[rows, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pX: cute.Pointer, pOut: cute.Pointer, B: Int32, L: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, K, V, C = self.H, self.K, self.V, self.C
        nc = L // C
        # output row (c, b, h, t) in order, so consecutive CTAs write consecutive rows
        t = bidx % C
        r1 = bidx // C
        h = r1 % H
        r2 = r1 // H
        b = r2 % B
        c = r2 // B
        src = pX + ((Int64(b * L + c * C + t) * K + h // self.n_rep) * V)
        dst = pOut + (Int64(bidx) * V)
        for i in cutlass.range_constexpr(V // 8 // 128 if V // 8 >= 128 else 1):
            e = (i * 128 + tidx) * 8
            if e < V:
                cute.make_tensor((dst + e).align(16), cute.make_layout(8)).store(
                    cute.make_tensor((src + e).align(16), cute.make_layout(8)).load())


class RowDot:
    """delta[b, h, l] = sum_v dout[b, l, h, v] y[b, l, h, v] in fp32 from the bf16 rows (one warp per row)."""

    def __init__(self, H: int, V: int):
        self.H, self.V = H, V
        assert V % 8 == 0

    @cute.jit
    def __call__(self, tDO: cute.Tensor, tY: cute.Tensor, tDelta: cute.Tensor, B: Int32, L: Int32, stream: cuda.CUstream):
        rows = B * L * self.H
        self.kernel(tDO.iterator, tY.iterator, tDelta.iterator, L, rows).launch(
            grid=[(rows + 3) // 4, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pDO: cute.Pointer, pY: cute.Pointer, pDelta: cute.Pointer, L: Int32, rows: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        H, V = self.H, self.V
        lane = tidx % 32
        row = bidx * 4 + tidx // 32  # (b, l, h) in order
        if row < rows:
            acc = Float32(0.0)
            base = Int64(row) * V
            for i in cutlass.range_constexpr((V // 8 + 31) // 32):
                e = (i * 32 + lane) * 8
                if e < V:
                    a = cute.make_tensor((pDO + (base + e)).align(16), cute.make_layout(8)).load()
                    c = cute.make_tensor((pY + (base + e)).align(16), cute.make_layout(8)).load()
                    for k in cutlass.range_constexpr(8):
                        acc = acc + a[k].to(Float32) * c[k].to(Float32)
            for off in cutlass.range_constexpr(5):
                acc = acc + cute.arch.shuffle_sync_bfly(acc, offset=1 << off)
            if lane == 0:
                h = row % H
                bl = row // H
                b = bl // L
                l = bl % L
                cute.make_tensor(pDelta + ((b * H + h) * L + l), cute.make_layout(1))[0] = acc


# ======================================================================================================
# Host side
# ======================================================================================================

def segment_starts(segment_idx: Optional[torch.Tensor], B: int, L: int, device) -> torch.Tensor:
    """kmin[b, t]: the first position of the segment of position t (0 without segments), int32 [B, L]."""
    if segment_idx is None:
        return torch.zeros(B, L, dtype=torch.int32, device=device)
    seg = segment_idx[:, :L]
    idx = torch.arange(L, device=device, dtype=torch.int32).expand(B, L)
    new = torch.ones(B, L, dtype=torch.bool, device=device)
    new[:, 1:] = seg[:, 1:] != seg[:, :-1]
    return torch.cummax(torch.where(new, idx, torch.zeros_like(idx)), dim=1).values.to(torch.int32)


def query_limits(kmin: torch.Tensor) -> torch.Tensor:
    """qmax[b, k]: the last query whose segment starts at or before key k (int32 [B, L]); kmin is nondecreasing."""
    B, L = kmin.shape
    pos = torch.arange(L, device=kmin.device, dtype=kmin.dtype).expand(B, L).contiguous()
    return (torch.searchsorted(kmin.contiguous(), pos, right=True) - 1).to(torch.int32)


def sca_bwd_dv(q, k, dout, lse2, kmin, qmax, chunk_size, scale):
    B, L, H, S = q.shape
    K, V = dout.shape[2] // (H // k.shape[2]) if False else k.shape[2], dout.shape[3]
    dv = torch.empty(B, L, K, V, dtype=torch.bfloat16, device=q.device)
    f = _compiled(("sca_bwd_dv", H, K, S, V, chunk_size), lambda: ScaBwdDv(H, K, S, V, chunk_size), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Float32, 4), _fake(Int32, 4), _fake(Int32, 4),
        _fake(BFloat16), Int32(0), Int32(0), Float32(0.0)))
    f(q.reshape(-1), k.reshape(-1), dout.reshape(-1), lse2.reshape(-1), kmin.reshape(-1), qmax.reshape(-1), dv.reshape(-1),
      B, L, float(scale * LOG2E))
    return dv


def sca_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int, scale: float,
            kmin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """q [B, L, H, S], k [B, L, K, S], v [B, L, K, V] (bf16, contiguous) -> y [B, L, H, V] bf16, lse2 [B, H, L] fp32."""
    B, L, H, S = q.shape
    K, V = v.shape[2], v.shape[3]
    C = chunk_size
    assert L % C == 0 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    dev = q.device
    y = torch.empty(B, L, H, V, dtype=torch.bfloat16, device=dev)
    lse2 = torch.empty(B, H, L, dtype=torch.float32, device=dev)
    f = _compiled(("sca_fwd", H, K, S, V, C), lambda: ScaFwd(H, K, S, V, C), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int32, 4), _fake(BFloat16), _fake(Float32, 4),
        Int32(0), Int32(0), Float32(0.0)))
    f(q.reshape(-1), k.reshape(-1), v.reshape(-1), kmin.reshape(-1), y.reshape(-1), lse2.reshape(-1), B, L,
      float(scale * LOG2E))
    return y, lse2


def sca_reference(q, k, v, chunk_size, scale, segment_idx=None, dtype=torch.float32):
    """Dense PyTorch reference (per chunk): y [B, L, H, V] and lse2 [B, H, L] (log2 domain, scaled)."""
    B, L, H, S = q.shape
    K, V = v.shape[2], v.shape[3]
    C = chunk_size
    nc = L // C
    n_rep = H // K
    qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
    kf = kf.repeat_interleave(n_rep, dim=2)
    vf = vf.repeat_interleave(n_rep, dim=2)
    y = torch.zeros(B, L, H, V, dtype=dtype, device=q.device)
    lse2 = torch.zeros(B, H, L, dtype=dtype, device=q.device)
    pos = torch.arange(L, device=q.device)
    for c in range(nc):
        qs, qe = c * C, (c + 1) * C
        ks = 0 if c == 0 else (c - 1) * C
        s = torch.einsum("bqhd,bkhd->bhqk", qf[:, qs:qe], kf[:, ks:qe]) * scale
        mask = pos[ks:qe][None, :] <= pos[qs:qe][:, None]
        if segment_idx is not None:
            mask = mask[None] & (segment_idx[:, ks:qe, None] == segment_idx[:, qs:qe]).transpose(1, 2)[:, None] if False else \
                mask[None] & (segment_idx[:, qs:qe, None] == segment_idx[:, None, ks:qe])
            s = s.masked_fill(~mask[:, None], float("-inf"))
        else:
            s = s.masked_fill(~mask[None, None], float("-inf"))
        m = s.amax(-1, keepdim=True)
        p = torch.exp(s - m)
        l = p.sum(-1, keepdim=True)
        y[:, qs:qe] = torch.einsum("bhqk,bkhv->bqhv", p / l, vf[:, ks:qe])
        lse2[:, :, qs:qe] = (m + torch.log(l)).squeeze(-1) * LOG2E
    return y, lse2
