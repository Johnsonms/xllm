"""Adaptive working memory (AWM) forward with CuTe DSL kernels for Blackwell (SM100).

The memory update of a chunk is linear in the memory state M (S x V):

    U = kk (v - qk M) = kk v - (kk qk) M,      M' = M + r * (U - M)

with kk = exp(k - lse)^T the softmax of the chunk's keys over time. So the forward runs in three phases:

  1. per update chunk j (parallel):  lse_j = logsumexp_t k_t,  A_j = kk_j qk_j (S x S),  B_j = kk_j v_j (S x V)
  2. per (batch, kv head) (sequential over chunks, tiny):  M_{c+1} = (1 - r) M_c + r (B_c - A_c M_c)
  3. per chunk c (parallel):  out_c = q_c M_c, rows outside the memory's segment zeroed

Phases 1 and 3 read every input once and write the output once; the large reductions over the chunk do not
depend on the recurrence. This file holds the kernels; see adaptive_working_memory_fwd for the host side.
"""

import math
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05


class AwmOutGemm:
    """Phase 3: out[b, l, h, :] = q[b, l, h, :] @ M_c(b, h // n_rep) for the chunk c of row l, zero on masked rows.

    q is read as [B, L, H, S] and out written as [B, L, H, V] (the layouts of the fused block, no transposes);
    the memory states come transposed and stacked as MT[(b * K + k) * nc + c, v, s].

    Persistent warp-specialized kernel: warps 0-3 epilogue (TMEM -> registers -> mask -> bf16 -> smem -> TMA
    store), warp 4 issues the tcgen05 MMAs, warp 5 the TMA loads. A tile is 128 rows x VT columns with the
    whole S as K. Tiles are ordered (vt, b, k, c, head of the group, row block) and every CTA takes a contiguous
    range, so the memory tile (VT x S, resident in shared memory) is reloaded only when the group changes.
    """

    def __init__(self, H: int, K: int, S: int, V: int, C: int, has_prev: bool, has_seg: bool, a_stages: int = 4):
        self.H, self.K, self.S, self.V, self.C = H, K, S, V, C
        self.has_prev, self.has_seg = has_prev, has_seg
        self.n_rep = H // K
        self.TM = 128
        self.VT = min(V, 256)
        self.KB = 64  # K block of one A stage (128 B of bf16: one swizzle atom)
        self.a_stages = a_stages
        self.acc_stages = 2
        self.epi_stages = 2
        self.mma_tiler = (self.TM, self.VT, self.KB)
        self.mma_tiler_b = (self.TM, self.VT, self.S)
        self.num_epi_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert H % K == 0 and V % self.VT == 0 and C % self.TM == 0 and S % self.KB == 0
        assert self.VT % 16 == 0 and self.VT <= 256
        self.n_vt = V // self.VT
        self.row_tiles = C // self.TM
        self.k_tiles = S // self.KB

    @cute.jit
    def __call__(self, tQ: cute.Tensor, tMT: cute.Tensor, tOut: cute.Tensor, tSeg: cute.Tensor, tPrevCount: cute.Tensor,
                 tDead: cute.Tensor, tOutMask: cute.Tensor, B: Int32, L: Int32, seg_off: Int32, seg_stride: Int32,
                 num_ctas: Int32, stream: cuda.CUstream):
        H, K, S, V = self.H, self.K, self.S, self.V
        nc = L // self.C
        pQ, pMT, pOut = tQ.iterator, tMT.iterator, tOut.iterator
        # q [B, L, H, S] as (M = L, K = S, H, B)
        mQ = cute.make_tensor(pQ, cute.make_layout((L, S, H, B), stride=(H * S, 1, S, L * H * S)))
        # MT [(b, k, c), V, S] as (N = V, K = S, (c, k, b))
        mMT = cute.make_tensor(pMT, cute.make_layout((V, S, B * K * nc), stride=(S, 1, V * S)))
        # out [B, L, H, V] as (M = L, N = V, H, B)
        mOut = cute.make_tensor(pOut, cute.make_layout((L, V, H, B), stride=(H * V, 1, V, L * H * V)))
        mOutMask = cute.make_tensor(tOutMask.iterator, cute.make_layout((L, B), stride=(1, L)))
        mDead = cute.make_tensor(tDead.iterator, cute.make_layout((nc, B), stride=(1, nc)))

        op = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, self.VT, 16), tcgen05.CtaGroup.ONE,
                                  tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
        tiled_mma = cute.make_tiled_mma(op)
        a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, BFloat16, self.a_stages)
        b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler_b, BFloat16, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mQ_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, mQ, cute.slice_(a_smem_layout, (None, None, None, 0)), self.mma_tiler, tiled_mma)
        tma_b, mMT_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, mMT, cute.slice_(b_smem_layout, (None, None, None, 0)), self.mma_tiler_b, tiled_mma)

        c_layout_kind = utils.LayoutEnum.ROW_MAJOR
        epi_tile = sm100_utils.compute_epilogue_tile_shape((self.TM, self.VT, self.KB), False, c_layout_kind, BFloat16)
        epi_smem_layout = sm100_utils.make_smem_layout_epi(BFloat16, c_layout_kind, epi_tile, self.epi_stages)
        tma_c, mOut_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mOut, cute.slice_(epi_smem_layout, (None, None, 0)), epi_tile)

        num_tiles = self.n_vt * B * K * nc * self.n_rep * self.row_tiles
        self.kernel(tiled_mma, tma_a, mQ_tma, tma_b, mMT_tma, tma_c, mOut_tma, tSeg.iterator, tPrevCount.iterator,
                    mDead, mOutMask, a_smem_layout, b_smem_layout, epi_smem_layout, epi_tile, c_layout_kind,
                    B, nc, num_tiles, seg_off, seg_stride).launch(grid=[num_ctas, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_a: cute.CopyAtom, mQ: cute.Tensor, tma_b: cute.CopyAtom,
               mMT: cute.Tensor, tma_c: cute.CopyAtom, mOut: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer,
               mDead: cute.Tensor, mOutMask: cute.Tensor, a_smem_layout: cute.ComposedLayout,
               b_smem_layout: cute.ComposedLayout, epi_smem_layout: cute.ComposedLayout, epi_tile: cute.Tile,
               c_layout_kind: cutlass.Constexpr, B: Int32, nc: Int32, num_tiles: Int32, seg_off: Int32,
               seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nctas, _, _ = cute.arch.grid_dim()

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
            cpasync.prefetch_descriptor(tma_c)

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.a_stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.acc_stages * 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        epi_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * self.num_epi_warps)
        tmem_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (self.num_epi_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)

        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_smem_layout, mode=[0, 1, 2]))
        b_bytes = cute.size_in_bytes(BFloat16, cute.select(b_smem_layout, mode=[0, 1, 2]))
        thread_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        a_prod, a_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(), num_stages=self.a_stages,
            producer_group=thread_group, consumer_group=thread_group, tx_count=a_bytes,
        ).make_participants()
        b_prod, b_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(), num_stages=1,
            producer_group=thread_group, consumer_group=thread_group, tx_count=b_bytes,
        ).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=self.acc_stages, producer_group=thread_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps),
        ).make_participants()

        sA = smem.allocate_tensor(BFloat16, a_smem_layout.outer, byte_alignment=128, swizzle=a_smem_layout.inner)
        sB = smem.allocate_tensor(BFloat16, b_smem_layout.outer, byte_alignment=128, swizzle=b_smem_layout.inner)
        sC = smem.allocate_tensor(BFloat16, epi_smem_layout.outer, byte_alignment=128, swizzle=epi_smem_layout.inner)

        # (bM, bK, RestM, RestK, H, B) / (bN, bK, RestN, RestK, NL) / (bM, bN, RestM, RestN, H, B)
        gA = cute.local_tile(mQ, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None, None))
        gB = cute.local_tile(mMT, cute.slice_(self.mma_tiler_b, (0, None, None)), (None, None, None))
        gC = cute.local_tile(mOut, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        tAsA, tAgA = cpasync.tma_partition(tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 3),
                                           cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3),
                                           cute.group_modes(tCgB, 0, 3))
        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None, None)], epi_tile)
        tCsC, tCgC_tma = cpasync.tma_partition(tma_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                                               cute.group_modes(gC_epi, 0, 2))

        # contiguous tile range of this CTA
        start = bidx * num_tiles // nctas
        end = (bidx + 1) * num_tiles // nctas
        group_tiles = self.n_rep * self.row_tiles

        if warp == self.tma_warp:
            prev_group = Int32(-1)
            t = start
            while t < end:
                rt, hr, c, k, b, vt = self._decode(t, B, nc)
                group = t // group_tiles
                if group != prev_group:
                    hb = b_prod.acquire_and_advance()
                    cute.copy(tma_b, tBgB[(None, vt, 0, (b * self.K + k) * nc + c)], tBsB[(None, hb.index)],
                              tma_bar_ptr=hb.barrier)
                    prev_group = group
                row_block = c * self.row_tiles + rt
                h = k * self.n_rep + hr
                for kt in cutlass.range(self.k_tiles):
                    ha = a_prod.acquire_and_advance()
                    cute.copy(tma_a, tAgA[(None, row_block, kt, h, b)], tAsA[(None, ha.index)], tma_bar_ptr=ha.barrier)
                t += 1
            a_prod.tail()
            b_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            t = start
            while t < end:
                group = t // group_tiles
                if t == start or t % group_tiles == 0:
                    b_cons.wait()
                acc_empty = acc_prod.acquire_and_advance()
                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kt in cutlass.range(self.k_tiles):
                    ha = a_cons.wait_and_advance()
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, ha.index)],
                                  tCrB[(None, None, kt * cute.size(tCrA, mode=[2]) + kb, 0)], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ha.release()
                acc_empty.commit()
                # release the memory tile after the group's last tile (the MMAs above still read it)
                if t + 1 == end or (t + 1) // group_tiles != group:
                    b_cons.release()
                    b_cons.advance()
                t += 1
            acc_prod.tail()
        elif warp < self.mma_warp:
            tmem.allocate(512)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            epi_pipe = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_epi_warps))
            copy_atom_t2r = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
            t = start
            while t < end:
                rt, hr, c, k, b, vt = self._decode(t, B, nc)
                row_block = c * self.row_tiles + rt
                h = k * self.n_rep + hr
                # this thread's row of the tile (TMEM lane = row for M = 128) and its mask: the chunk's memory is
                # empty, or the row is not in the segment that the memory ends in (that of chunk c - 2)
                row = row_block * self.TM + tidx
                masked = mDead[c, b] != 0
                if const_expr(self.has_seg):
                    if c >= 2 or const_expr(self.has_prev):
                        seg_row = cute.make_tensor(pSeg + (b * seg_stride + seg_off + row), cute.make_layout(1))[0]
                        masked = masked | (seg_row != _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C))
                keep = cutlass.select_(masked, Float32(0.0), Float32(1.0))
                if hr == 0 and vt == 0 and k == 0:
                    mOutMask[row, b] = cutlass.select_(masked, Uint8(1), Uint8(0))
                acc_full = acc_cons.wait_and_advance()
                tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]
                tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
                tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, row_block, vt, h, b)], epi_tile)
                tCgC_tma_cur = tCgC_tma[(None, None, None, row_block, vt, h, b)]
                tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)])
                thr_t2r = tiled_t2r.get_slice(tidx)
                tTR_tAcc = thr_t2r.partition_S(tCtAcc_epi)
                tTR_gC = thr_t2r.partition_D(tCgC_epi)
                tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, Float32)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                copy_atom_r2s = sm100_utils.get_smem_store_op(c_layout_kind, BFloat16, Float32, tiled_t2r)
                tiled_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_t2r)
                thr_r2s = tiled_r2s.get_slice(tidx)
                tRS_sC = thr_r2s.partition_D(sC)
                tRS_rAcc = tiled_r2s.retile(tTR_rAcc)
                tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, BFloat16)
                tCgC_grouped = cute.group_modes(tCgC_tma_cur, 1, cute.rank(tCgC_tma_cur))
                for sub in cutlass.range(cute.size(tTR_tAcc.shape, mode=[3])):
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, sub)], tTR_rAcc)
                    buf = sub % self.epi_stages
                    tRS_rC.store((tRS_rAcc.load() * keep).to(BFloat16))
                    cute.copy(tiled_r2s, tRS_rC, tRS_sC[(None, None, None, buf)])
                    cute.arch.fence_view_async_shared()
                    epi_bar.arrive_and_wait()
                    if warp == 0:
                        cute.copy(tma_c, tCsC[(None, buf)], tCgC_grouped[(None, sub)])
                        epi_pipe.producer_commit()
                        epi_pipe.producer_acquire()
                    epi_bar.arrive_and_wait()
                with cute.arch.elect_one():
                    acc_full.release()
                t += 1
            epi_pipe.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def _decode(self, t: Int32, B: Int32, nc: Int32):
        """Tile index -> (row block in chunk, head in group, chunk, kv head, batch, v tile)."""
        rt = t % self.row_tiles
        t1 = t // self.row_tiles
        hr = t1 % self.n_rep
        t2 = t1 // self.n_rep
        c = t2 % nc
        t3 = t2 // nc
        k = t3 % self.K
        t4 = t3 // self.K
        b = t4 % B
        vt = t4 // B
        return rt, hr, c, k, b, vt


LOG2E = math.log2(math.e)
NEG_BIG = -1.0e30  # finite stand-in for -inf in running maxima (exp2 of differences stays 0, never NaN)


class AwmKeyLsePart:
    """Phase 1a: partial base-2 log-sum-exp of the keys over time. For update chunk j (output index
    j + j_off) and row split rs (C / R rows), part[b, k, j + j_off, rs, s] = (m, l) with
    log2 sum_t 2^(k[t, s] log2 e) = m + log2 l over the split's rows; rows outside the segment of their chunk's
    last row are excluded (segment ids read from seg[b * seg_stride + seg_off + t] when has_seg).
    The consumers combine the R partials. k is read as [B, L, K, S].

    One CTA per (b, k, j, 64 columns of s, row split), 256 threads = 8 column groups (8 columns, one 16-byte
    load each) x 32 row groups; online in fp32, rescaled once per 4 rows; row groups combined in shared memory."""

    def __init__(self, K: int, S: int, C: int, R: int, has_seg: bool):
        self.K, self.S, self.C, self.R, self.has_seg = K, S, C, R, has_seg
        self.SB = 64
        self.RG = 32
        self.UNROLL = 4
        assert S % self.SB == 0 and C % (R * self.RG * self.UNROLL) == 0

    @cute.jit
    def __call__(self, tK: cute.Tensor, tSeg: cute.Tensor, tPart: cute.Tensor, B: Int32, L: Int32, nu: Int32,
                 j_off: Int32, nu_tot: Int32, seg_off: Int32, seg_stride: Int32, stream: cuda.CUstream):
        K, S, R = self.K, self.S, self.R
        mK = cute.make_tensor(tK.iterator, cute.make_layout((S, L, K, B), stride=(1, K * S, S, L * K * S)))
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu_tot, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu_tot, 2 * S * R * nu_tot * K)))
        grid = B * K * nu * (S // self.SB) * R
        self.kernel(mK, tSeg.iterator, mPart, nu, j_off, seg_off, seg_stride).launch(
            grid=[grid, 1, 1], block=[8 * self.RG, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mK: cute.Tensor, pSeg: cute.Pointer, mPart: cute.Tensor, nu: Int32, j_off: Int32,
               seg_off: Int32, seg_stride: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nsb = self.S // self.SB
        rs = bidx % self.R
        rest = bidx // self.R
        sb = rest % nsb
        rest = rest // nsb
        j = rest % nu
        rest = rest // nu
        k = rest % self.K
        b = rest // self.K
        sg = tidx % 8
        rg = tidx // 8
        s0 = sb * self.SB + sg * 8
        rows = self.C // self.R
        seg_last = Int64(0)
        if const_expr(self.has_seg):
            seg_last = cute.make_tensor(pSeg + (b * seg_stride + seg_off + j * self.C + self.C - 1), cute.make_layout(1))[0]
        m = [Float32(NEG_BIG) for _ in range(8)]
        acc = [Float32(0.0) for _ in range(8)]
        r_base = j * self.C + rs * rows + rg
        for it in cutlass.range(rows // (self.RG * self.UNROLL)):
            xs = []
            for u in cutlass.range_constexpr(self.UNROLL):
                r = r_base + (it * self.UNROLL + u) * self.RG
                v = cute.make_tensor((mK.iterator + mK.layout((s0, r, k, b))).align(16), cute.make_layout(8)).load().to(Float32)
                if const_expr(self.has_seg):
                    masked = cute.make_tensor(pSeg + (b * seg_stride + seg_off + r), cute.make_layout(1))[0] != seg_last
                    xs.append([cutlass.select_(masked, Float32(NEG_BIG), v[e] * LOG2E) for e in range(8)])
                else:
                    xs.append([v[e] * LOG2E for e in range(8)])
            for e in cutlass.range_constexpr(8):
                bmax = xs[0][e]
                for u in cutlass.range_constexpr(1, self.UNROLL):
                    bmax = cute.arch.fmax(bmax, xs[u][e])
                m_new = cute.arch.fmax(m[e], bmax)
                a = acc[e] * cute.math.exp2(m[e] - m_new, fastmath=True)
                for u in cutlass.range_constexpr(self.UNROLL):
                    a = a + cute.math.exp2(xs[u][e] - m_new, fastmath=True)
                acc[e] = a
                m[e] = m_new
        smem = utils.SmemAllocator()
        sM = smem.allocate_tensor(Float32, cute.make_layout((self.SB, self.RG), stride=(1, self.SB)), byte_alignment=16)
        sL = smem.allocate_tensor(Float32, cute.make_layout((self.SB, self.RG), stride=(1, self.SB)), byte_alignment=16)
        for e in cutlass.range_constexpr(8):
            sM[sg * 8 + e, rg] = m[e]
            sL[sg * 8 + e, rg] = acc[e]
        cute.arch.sync_threads()
        if tidx < self.SB:
            mm = sM[tidx, 0]
            for g in cutlass.range_constexpr(1, self.RG):
                mm = cute.arch.fmax(mm, sM[tidx, g])
            tot = Float32(0.0)
            for g in cutlass.range_constexpr(self.RG):
                tot = tot + sL[tidx, g] * cute.math.exp2(sM[tidx, g] - mm, fastmath=True)
            mPart[0, sb * self.SB + tidx, rs, j + j_off, k, b] = mm
            mPart[1, sb * self.SB + tidx, rs, j + j_off, k, b] = tot


@cute.jit
def _seg_anchor(pSeg: cute.Pointer, pPrevCount: cute.Pointer, b, x, seg_off, seg_stride, C: cutlass.Constexpr):
    """Segment id at the end of chunk x: the last row of chunk x for x >= 0, of the previous call's last chunk for
    x = -1 (stored in front of the current rows), the previous segment count for x = -2."""
    val = Int64(0)
    if x == -2:
        val = cute.make_tensor(pPrevCount + b, cute.make_layout(1))[0]
    else:
        val = cute.make_tensor(pSeg + (b * seg_stride + seg_off + x * C + C - 1), cute.make_layout(1))[0]
    return val


@cute.jit
def _combine_lse2(mPart: cute.Tensor, s, j, k, b, R: cutlass.Constexpr):
    """log2 sum over the R partials (m, l) of row s: max m + log2 sum l 2^(m - max m)."""
    mm = mPart[0, s, 0, j, k, b]
    for r in cutlass.range_constexpr(1, R):
        mm = cute.arch.fmax(mm, mPart[0, s, r, j, k, b])
    tot = Float32(0.0)
    for r in cutlass.range_constexpr(R):
        tot = tot + mPart[1, s, r, j, k, b] * cute.math.exp2(mPart[0, s, r, j, k, b] - mm, fastmath=True)
    return mm + cute.math.log2(tot, fastmath=True)


def chunk_gemm_tiling(S: int, V: int) -> Tuple[int, int, int]:
    """(NS, nsub, n_blocks): the N = S + V columns of [qk | v] in sub-tiles of NS columns (one MMA each),
    nsub sub-tiles per CTA (nsub * NS <= 512 TMEM columns, nsub dividing the number of sub-tiles so that every
    CTA loads the same number of bytes per stage). Fewest CTAs per output row block, then the widest MMA."""
    best = None
    for ns in (256, 128, 64):
        if S % ns or V % ns:
            continue
        total = (S + V) // ns
        nsub = max(d for d in range(1, 512 // ns + 1) if total % d == 0)
        cand = (total // nsub, -ns, nsub)
        if best is None or cand < best:
            best = cand
    nb, neg_ns, nsub = best
    return -neg_ns, nsub, nb


class AwmChunkGemm:
    """Phase 1b: AB[b, k, j, s, :] += sum_t P[s, t] [qk | v][t, :] over the rows t of a split of update chunk j,
    P[s, t] = exp(k[b, t, k, s] - lse[b, k, j, s]) (0 on masked rows). AB is fp32, zeroed by the caller.

    Both MMA operands are MN-major (k, qk and v are read as [B, L, K, *], the s / n columns contiguous).
    One CTA per (b, k, j, 128 rows of s, block of nsub sub-tiles of NS columns, split of the chunk):
      warp 5:    TMA loads of the raw k tile (128 s x KB rows) and the nsub [qk | v] sub-tiles of a stage
      warps 0-3: P = exp2(k * log2e - lse * log2e) of the stage into a second buffer (16-byte vectors at the
                 same swizzled addresses as the raw tile, the 8 lse values of a thread held in registers);
                 at the end, the epilogue (TMEM -> registers -> red.global.add.v4.f32 into AB)
      warp 4:    tcgen05 MMAs, one fp32 accumulator of NS TMEM columns per sub-tile
    """

    def __init__(self, K: int, S: int, V: int, C: int, splits: int, R: int, has_seg: bool, stages: int = 4):
        self.K, self.S, self.V, self.C, self.R, self.has_seg = K, S, V, C, R, has_seg
        self.NS, self.nsub, self.n_blocks = chunk_gemm_tiling(S, V)
        self.TM = 128
        self.KB = 32
        self.stages = stages
        self.splits = splits
        self.m_blocks = S // self.TM
        self.chunk_k_tiles = C // self.KB  # split as evenly as possible among the splits
        self.qk_subs = S // self.NS
        self.NAB = S + V
        self.num_tp_warps = 4
        self.mma_warp = 4
        self.tma_warp = 5
        self.threads = 32 * 6
        assert S % self.TM == 0 and C % self.KB == 0 and splits <= self.chunk_k_tiles
        assert self.nsub * self.NS <= 512

    @cute.jit
    def __call__(self, tK: cute.Tensor, tQK: cute.Tensor, tV: cute.Tensor, tSeg: cute.Tensor, tPart: cute.Tensor,
                 tAB: cute.Tensor, B: Int32, L: Int32, nu: Int32, j_off: Int32, nu_tot: Int32, seg_off: Int32,
                 seg_stride: Int32, stream: cuda.CUstream):
        K, S, V = self.K, self.S, self.V
        # operands as (MN, K = time, kv head, batch), MN contiguous
        mK = cute.make_tensor(tK.iterator, cute.make_layout((S, L, K, B), stride=(1, K * S, S, L * K * S)))
        mQK = cute.make_tensor(tQK.iterator, cute.make_layout((S, L, K, B), stride=(1, K * S, S, L * K * S)))
        mV = cute.make_tensor(tV.iterator, cute.make_layout((V, L, K, B), stride=(1, K * V, V, L * K * V)))
        R = self.R
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu_tot, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu_tot, 2 * S * R * nu_tot * K)))
        mAB = cute.make_tensor(tAB.iterator, cute.make_layout((self.NAB, S, nu_tot, K, B), stride=(
            1, self.NAB, S * self.NAB, nu_tot * S * self.NAB, K * nu_tot * S * self.NAB)))

        op = tcgen05.MmaF16BF16Op(BFloat16, Float32, (self.TM, self.NS, 16), tcgen05.CtaGroup.ONE,
                                  tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN)
        tiled_mma = cute.make_tiled_mma(op)
        tiler = (self.TM, self.NS, self.KB)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, tiler, BFloat16, self.stages)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, tiler, BFloat16, self.stages * self.nsub)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_A(tma_op, mK, cute.slice_(a_layout, (None, None, None, 0)), tiler, tiled_mma)
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_qk, mQK_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mQK, b_one, tiler, tiled_mma)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(tma_op, mV, b_one, tiler, tiled_mma)
        grid = B * K * nu * self.m_blocks * self.n_blocks * self.splits
        self.kernel(tiled_mma, tma_k, mK_t, tma_qk, mQK_t, tma_v, mV_t, tSeg.iterator, mPart, mAB, a_layout, b_layout,
                    B, nu, j_off, seg_off, seg_stride).launch(grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_k: cute.CopyAtom, mK: cute.Tensor, tma_qk: cute.CopyAtom,
               mQK: cute.Tensor, tma_v: cute.CopyAtom, mV: cute.Tensor, pSeg: cute.Pointer, mPart: cute.Tensor,
               mAB: cute.Tensor, a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout, B: Int32, nu: Int32,
               j_off: Int32, seg_off: Int32, seg_stride: Int32):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # unit decode: (split, n block, m block, j, k, b)
        ks = bidx % self.splits
        rest = bidx // self.splits
        nb = rest % self.n_blocks
        rest = rest // self.n_blocks
        mb = rest % self.m_blocks
        rest = rest // self.m_blocks
        j = rest % nu
        rest = rest // nu
        k = rest % self.K
        b = rest // self.K
        kt_lo = ks * self.chunk_k_tiles // self.splits
        n_kt = (ks + 1) * self.chunk_k_tiles // self.splits - kt_lo
        row0 = j * self.C + kt_lo * self.KB  # first time row of this split
        kt0 = row0 // self.KB

        if warp == self.tma_warp:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_qk)
            cpasync.prefetch_descriptor(tma_v)

        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            p_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * (self.num_tp_warps + 1))
        tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=tmem_bar, allocator_warp_id=0)

        a_bytes = cute.size_in_bytes(BFloat16, cute.select(a_layout, mode=[0, 1, 2]))
        b_bytes = cute.size_in_bytes(BFloat16, cute.select(b_layout, mode=[0, 1, 2])) * self.nsub
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        tp_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * self.num_tp_warps)
        # the transform warps release a raw stage with one arrive per warp (lane 0, after a warp sync)
        k_prod, k_cons = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar.data_ptr(), num_stages=self.stages, producer_group=one,
            consumer_group=tp_group, tx_count=a_bytes, tidx=tidx % 32,
            enable_multicast_signaling=True).make_participants()
        p_prod, p_cons = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mbar.data_ptr(), num_stages=self.stages, producer_group=tp_group,
            consumer_group=one).make_participants()
        b_prod, b_cons = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(), num_stages=self.stages, producer_group=one,
            consumer_group=one, tx_count=b_bytes).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(), num_stages=1, producer_group=one,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_tp_warps)).make_participants()

        a_elems = cute.cosize(a_layout.outer)
        pRaw = smem.allocate_array(BFloat16, a_elems, byte_alignment=1024)
        pP = smem.allocate_array(BFloat16, a_elems, byte_alignment=1024)
        sRaw = cute.make_tensor(cute.recast_ptr(pRaw, swizzle_=a_layout.inner), a_layout.outer)
        sP = cute.make_tensor(cute.recast_ptr(pP, swizzle_=a_layout.inner), a_layout.outer)
        sB = smem.allocate_tensor(BFloat16, b_layout.outer, byte_alignment=1024, swizzle=b_layout.inner)

        # (bM, bK, RestM, RestK, K, B) / (bN, bK, RestN, RestK, K, B)
        gK = cute.local_tile(mK, (self.TM, self.KB), (None, None, None, None))
        gQK = cute.local_tile(mQK, (self.NS, self.KB), (None, None, None, None))
        gV = cute.local_tile(mV, (self.NS, self.KB), (None, None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgK = thr_mma.partition_A(gK)
        tCgQK = thr_mma.partition_B(gQK)
        tCgV = thr_mma.partition_B(gV)
        tCrA = tiled_mma.make_fragment_A(sP)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((self.TM, self.NS))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.nsub))
        tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sRaw, 0, 3), cute.group_modes(tCgK, 0, 3))
        tQKs, tQKg = cpasync.tma_partition(tma_qk, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgQK, 0, 3))
        tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sB, 0, 3), cute.group_modes(tCgV, 0, 3))

        if warp == self.tma_warp:
            for t in cutlass.range(n_kt):
                hk = k_prod.acquire_and_advance()
                cute.copy(tma_k, tKg[(None, mb, kt0 + t, k, b)], tKs[(None, hk.index)], tma_bar_ptr=hk.barrier)
                hb = b_prod.acquire_and_advance()
                for sub in cutlass.range_constexpr(self.nsub):
                    gsub = nb * self.nsub + sub
                    if gsub < self.qk_subs:
                        cute.copy(tma_qk, tQKg[(None, gsub, kt0 + t, k, b)], tQKs[(None, hb.index * self.nsub + sub)],
                                  tma_bar_ptr=hb.barrier)
                    else:
                        cute.copy(tma_v, tVg[(None, gsub - self.qk_subs, kt0 + t, k, b)], tVs[(None, hb.index * self.nsub + sub)],
                                  tma_bar_ptr=hb.barrier)
            k_prod.tail()
            b_prod.tail()
        elif warp == self.mma_warp:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_empty = acc_prod.acquire_and_advance()
            for t in cutlass.range(n_kt):
                hp = p_cons.wait_and_advance()
                hb = b_cons.wait_and_advance()
                for sub in cutlass.range_constexpr(self.nsub):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, t > 0)
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(tiled_mma, tCtAcc[(None, None, None, sub)], tCrA[(None, None, kb, hp.index)],
                                  tCrB[(None, None, kb, hb.index * self.nsub + sub)], tCtAcc[(None, None, None, sub)])
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                hp.release()
                hb.release()
            acc_empty.commit()
            acc_prod.tail()
        elif warp < self.mma_warp:
            tmem.allocate(512)
            # every TMEM user passes the allocation barrier before its main loop (the MMA warp waits there)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            # ---- transform: P = exp2(k * log2e - lse * log2e), 0 on masked rows ----
            g = tidx % 16  # 8 consecutive s rows: 8g .. 8g + 7 of the CTA's 128
            kq = tidx // 16  # time rows kq + 8 r of a stage
            s_base = mb * self.TM + 8 * g
            lse2 = [_combine_lse2(mPart, s_base + e, j + j_off, k, b, self.R) for e in range(8)]
            seg_row0 = b * seg_stride + seg_off + row0
            seg_last = Int64(0)
            if const_expr(self.has_seg):
                seg_last = cute.make_tensor(pSeg + (b * seg_stride + seg_off + j * self.C + self.C - 1), cute.make_layout(1))[0]
            stage_elems = self.TM * self.KB
            for t in cutlass.range(n_kt):
                hk = k_cons.wait_and_advance()
                hp = p_prod.acquire_and_advance()
                for r in cutlass.range_constexpr(self.KB // 8):
                    kk = kq + 8 * r
                    keep = Float32(1.0)
                    if const_expr(self.has_seg):
                        seg_r = cute.make_tensor(pSeg + (seg_row0 + t * self.KB + kk), cute.make_layout(1))[0]
                        keep = cutlass.select_(seg_r != seg_last, Float32(0.0), Float32(1.0))
                    off = ((8 * g) % 64 + ((8 * g) // 64) * (64 * self.KB) + (kk % 16) * 64 + (kk // 16) * 1024) * 2
                    off = off ^ (((off >> 7) & 7) << 4)
                    elem = hk.index * stage_elems + off // 2
                    x = cute.make_tensor((pRaw + elem).align(16), cute.make_layout(8)).load().to(Float32)
                    pv = cute.make_rmem_tensor((8,), Float32)
                    for e in cutlass.range_constexpr(8):
                        pv[e] = cute.math.exp2(x[e] * LOG2E - lse2[e], fastmath=True) * keep
                    cute.make_tensor((pP + (hp.index * stage_elems + off // 2)).align(16), cute.make_layout(8)).store(pv.load().to(BFloat16))
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                hk.release()
                hp.commit()
            # ---- epilogue: TMEM -> registers -> red.add into AB (this thread's s row) ----
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_full = acc_cons.wait_and_advance()
            copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE), Float32)
            s_row = mb * self.TM + tidx
            for sub in cutlass.range_constexpr(self.nsub):
                acc_sub = tCtAcc[((None, None), 0, 0, sub)]
                acc_epi = cute.flat_divide(acc_sub, (self.TM, 32))
                tiled_t2r = tcgen05.make_tmem_copy(copy_atom, acc_epi[(None, None, 0, 0)])
                thr_t2r = tiled_t2r.get_slice(tidx)
                tTR = thr_t2r.partition_S(acc_epi)
                regs = cute.make_rmem_tensor(thr_t2r.partition_D(cute.make_identity_tensor((self.TM, 32))).shape, Float32)
                col_base = (nb * self.nsub + sub) * self.NS
                for cchunk in cutlass.range_constexpr(self.NS // 32):
                    cute.copy(tiled_t2r, tTR[(None, None, None, 0, cchunk)], regs)
                    for q4 in cutlass.range_constexpr(8):
                        quad = cute.make_rmem_tensor((4,), Float32)
                        for e in cutlass.range_constexpr(4):
                            quad[e] = regs[q4 * 4 + e]
                        dst = mAB.iterator + mAB.layout((col_base + cchunk * 32 + q4 * 4, s_row, j + j_off, k, b))
                        cute.arch.red(dst.llvm_ptr, quad.load(), op="add", dtype=Float32, sem="relaxed", scope="gpu")
            with cute.arch.elect_one():
                acc_full.release()
            tmem.relinquish_alloc_permit()
            cute.arch.barrier(barrier_id=2, number_of_threads=32 * self.num_tp_warps)
            tmem.free(tmem_ptr)


class AwmRecurrence:
    """Phase 2: per (b, kv head) the memory over the chunks:

        M_c is the state used by chunk c; the update at chunk c (c >= first) uses AB / lse of update index c - first:
        M, lnt <- 0, -inf where memory_mask[b, c];   lnt' = logaddexp(lse, lnt);   r = exp(lse - lnt')
        M <- (1 - r) M + r (B - A M)

    Writes MT[b, k, c] = M_c^T (bf16, the B operand of phase 3), the final memory / log normalizer, and marks
    the rows of chunks whose memory is still empty (lnt[b, 0, 0] = -inf) in the output mask.

    One CTA (8 warps) per (b, k, VT columns of V). M stays in fp32 in the warp-MMA accumulator layout; the
    product A M runs on tensor cores (mma.sync, bf16 operands, fp32 accumulation) with A staged in shared memory
    KB columns at a time and M^T kept in shared memory as bf16."""

    def __init__(self, K: int, S: int, V: int, C: int, R: int, has_memory: bool, has_prev: bool, has_seg: bool,
                 VT: int = 64):
        self.K, self.S, self.V, self.C, self.R = K, S, V, C, R
        self.has_memory, self.has_prev, self.has_seg = has_memory, has_prev, has_seg
        self.VT = min(VT, V)
        self.KB = 64
        self.NAB = S + V
        self.threads = 256
        self.PAD = 8  # bf16 elements of row padding in shared memory (bank conflicts)
        assert V % self.VT == 0 and S % self.KB == 0 and S % 128 == 0 and self.VT % 16 == 0

    @cute.jit
    def __call__(self, tPart: cute.Tensor, tAB: cute.Tensor, tMemIn: cute.Tensor, tLntIn: cute.Tensor,
                 tSeg: cute.Tensor, tPrevCount: cute.Tensor, tDead: cute.Tensor, tMT: cute.Tensor, tMemOut: cute.Tensor,
                 tLntOut: cute.Tensor, B: Int32, nc: Int32, nu: Int32, seg_off: Int32, seg_stride: Int32,
                 stream: cuda.CUstream):
        K, S, V = self.K, self.S, self.V
        R = self.R
        mPart = cute.make_tensor(tPart.iterator, cute.make_layout((2, S, R, nu, K, B), stride=(
            1, 2, 2 * S, 2 * S * R, 2 * S * R * nu, 2 * S * R * nu * K)))
        mAB = cute.make_tensor(tAB.iterator, cute.make_layout((self.NAB, S, nu, K, B), stride=(
            1, self.NAB, S * self.NAB, nu * S * self.NAB, K * nu * S * self.NAB)))
        mMemIn = cute.make_tensor(tMemIn.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mLntIn = cute.make_tensor(tLntIn.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))
        mDead = cute.make_tensor(tDead.iterator, cute.make_layout((nc, B), stride=(1, nc)))
        mMT = cute.make_tensor(tMT.iterator, cute.make_layout((S, V, nc, K, B), stride=(1, S, V * S, nc * V * S, K * nc * V * S)))
        mMemOut = cute.make_tensor(tMemOut.iterator, cute.make_layout((V, S, K, B), stride=(1, V, S * V, K * S * V)))
        mLntOut = cute.make_tensor(tLntOut.iterator, cute.make_layout((S, K, B), stride=(1, S, K * S)))
        op = cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((8, 1, 1)), permutation_mnk=(128, 16, 16))
        grid = B * K * (V // self.VT)
        self.kernel(tiled_mma, mPart, mAB, mMemIn, mLntIn, tSeg.iterator, tPrevCount.iterator, mDead, mMT, mMemOut,
                    mLntOut, nc, seg_off, seg_stride).launch(
            grid=[grid, 1, 1], block=[self.threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, mPart: cute.Tensor, mAB: cute.Tensor, mMemIn: cute.Tensor,
               mLntIn: cute.Tensor, pSeg: cute.Pointer, pPrevCount: cute.Pointer, mDead: cute.Tensor, mMT: cute.Tensor,
               mMemOut: cute.Tensor, mLntOut: cute.Tensor, nc: Int32, seg_off: Int32, seg_stride: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        S, VT, KB = self.S, self.VT, self.KB
        n_vt = self.V // VT
        vt = bidx % n_vt
        k = (bidx // n_vt) % self.K
        b = bidx // (n_vt * self.K)
        v0 = vt * VT
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, cute.make_layout((S, KB), stride=(KB + self.PAD, 1)), byte_alignment=16)
        sMT = smem.allocate_tensor(BFloat16, cute.make_layout((VT, S), stride=(S + self.PAD, 1)), byte_alignment=16)
        sR = smem.allocate_tensor(Float32, cute.make_layout(S), byte_alignment=16)

        thr_mma = tiled_mma.get_slice(tidx)
        coords = thr_mma.partition_C(cute.make_identity_tensor((S, VT)))
        m = cute.make_rmem_tensor(coords.shape, Float32)
        acc = cute.make_rmem_tensor(coords.shape, Float32)
        n_frag = cute.size(coords)
        # initial memory (fp32 copy in registers), log normalizer of row s held by thread s
        if const_expr(self.has_memory):
            for i in cutlass.range_constexpr(n_frag):
                s1, v1 = coords[i]
                m[i] = mMemIn[v0 + v1, s1, k, b].to(Float32)
        else:
            m.fill(0.0)
        lnt = Float32(-math.inf)
        if tidx < S:
            if const_expr(self.has_memory):
                lnt = mLntIn[tidx, k, b]

        first = 0 if const_expr(self.has_prev) else 1  # first chunk with an update
        for c in cutlass.range(nc):
            for i in cutlass.range_constexpr(n_frag):
                s2, v2 = coords[i]
                mMT[s2, v0 + v2, c, k, b] = m[i].to(BFloat16)
            # outputs of a chunk whose memory is still empty are masked (lnt[b, 0, 0] = -inf)
            if k == 0 and vt == 0 and tidx == 0:
                mDead[c, b] = cutlass.select_(lnt == Float32(-math.inf), Uint8(1), Uint8(0))
            if c >= first:
                u = c - first
                reset = False
                if const_expr(self.has_seg):
                    # the memory restarts when the chunks before the update's chunk end in different segments
                    if c >= 2 or const_expr(self.has_prev):
                        reset = _seg_anchor(pSeg, pPrevCount, b, c - 2, seg_off, seg_stride, self.C) != \
                            _seg_anchor(pSeg, pPrevCount, b, c - 1, seg_off, seg_stride, self.C)
                if reset:
                    m.fill(0.0)
                if tidx < S:
                    if reset:
                        lnt = Float32(-math.inf)
                    l_cur = _combine_lse2(mPart, tidx, u, k, b, self.R) * (1.0 / LOG2E)
                    hi = cute.arch.fmax(l_cur, lnt)
                    new_lnt = hi + cute.math.log(cute.math.exp(l_cur - hi) + cute.math.exp(lnt - hi))
                    sR[tidx] = cute.math.exp(l_cur - new_lnt)
                    lnt = new_lnt
                cute.arch.barrier()
                # M^T (bf16) into shared memory: the B operand
                for i in cutlass.range_constexpr(n_frag):
                    s3, v3 = coords[i]
                    sMT[v3, s3] = m[i].to(BFloat16)
                acc.fill(0.0)
                for kb in cutlass.range(S // KB):
                    cute.arch.barrier()
                    # A[:, kb * KB : (kb + 1) * KB] (fp32, rows contiguous) -> bf16 shared memory
                    for it in cutlass.range_constexpr(S * KB // self.threads):
                        idx = it * self.threads + tidx
                        row = idx // KB
                        col = idx % KB
                        sA[row, col] = mAB[kb * KB + col, row, u, k, b].to(BFloat16)
                    cute.arch.barrier()
                    tCsA = thr_mma.partition_A(sA)
                    tCsB = thr_mma.partition_B(cute.local_tile(sMT, (VT, KB), (0, kb)))
                    tCrA = tiled_mma.make_fragment_A(tCsA)
                    tCrB = tiled_mma.make_fragment_B(tCsB)
                    cute.autovec_copy(tCsA, tCrA)
                    cute.autovec_copy(tCsB, tCrB)
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cute.gemm(tiled_mma, acc, tCrA[None, None, kk], tCrB[None, None, kk], acc)
                for i in cutlass.range_constexpr(n_frag):
                    s4, v4 = coords[i]
                    r = sR[s4]
                    m[i] = (1.0 - r) * m[i] + r * (mAB[S + v0 + v4, s4, u, k, b] - acc[i])
                cute.arch.barrier()  # sMT / sR are rewritten in the next chunk
        for i in cutlass.range_constexpr(n_frag):
            s5, v5 = coords[i]
            mMemOut[v0 + v5, s5, k, b] = m[i].to(BFloat16)
        if vt == 0 and tidx < S:
            mLntOut[tidx, k, b] = lnt


# ======================================================================================================
# Host side
# ======================================================================================================

import torch  # noqa: E402

_COMPILED = {}


def _fake(dtype, align=16):
    return cute.runtime.make_fake_compact_tensor(dtype, (cute.sym_int(),), assumed_align=align)


def _fake_stream():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


def _compiled(key, make_op, make_args):
    """The compiled kernel for ``key`` (fake arguments are only built on the first call)."""
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = cute.compile(make_op(), *make_args(), _fake_stream(), options="--enable-tvm-ffi")
    return fn


_U8 = cutlass.Uint8


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _blhs(t: torch.Tensor) -> torch.Tensor:
    """[B, H, L, D] view -> [B, L, H, D] contiguous storage (free for the transposed views of the fused block)."""
    t = t.transpose(1, 2)
    return t if t.is_contiguous() else t.contiguous()


_DUMMY = {}


def _dummy(dtype, device):
    """A 1-element tensor standing in for an absent optional input (never read)."""
    key = (dtype, device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(1, dtype=dtype, device=device)
    return _DUMMY[key]


def _chunk_ab(k, qk, v, seg, part, AB, K, S, V, C, R, nu, j_off, seg_off, has_seg):
    """Phase 1 on [B, L, K, *] tensors for update chunks 0..nu-1 (written at chunk index j_off + j of the
    workspaces): lse partials [B, K, nu_tot, R, S, 2] and AB [B, K, nu_tot, S, S + V] (fp32, zeroed)."""
    B, L = k.shape[0], k.shape[1]
    nu_tot = part.shape[2]
    seg_stride = seg.shape[1] if has_seg else 0
    f_lse = _compiled(("lse", K, S, C, R, has_seg), lambda: AwmKeyLsePart(K, S, C, R, has_seg), lambda: (
        _fake(BFloat16), _fake(Int64, 8), _fake(Float32), *([Int32(0)] * 7)))
    f_lse(k.reshape(-1), seg.reshape(-1), part.reshape(-1), B, L, nu, j_off, nu_tot, seg_off, seg_stride)
    _, _, n_blocks = chunk_gemm_tiling(S, V)
    base = B * K * nu * (S // 128) * n_blocks
    splits = max(1, min(C // 32, round(_sm_count(k.device) / base)))
    f_ab = _compiled(("ab", K, S, V, C, splits, R, has_seg), lambda: AwmChunkGemm(K, S, V, C, splits, R, has_seg),
                     lambda: (_fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int64, 8), _fake(Float32),
                              _fake(Float32), *([Int32(0)] * 7)))
    f_ab(k.reshape(-1), qk.reshape(-1), v.reshape(-1), seg.reshape(-1), part.reshape(-1), AB.reshape(-1),
         B, L, nu, j_off, nu_tot, seg_off, seg_stride)


def adaptive_working_memory_fwd(
    query, qkey, key, value, chunk_size, memory, log_norm_term, prev_qkey, prev_key, prev_value,
    segment_idx=None, prev_segment_count=None, ortho=False, eps=1e-6,
):
    """Drop-in for xllm.modules.fused_ops.adaptive_working_memory._adaptive_working_memory_fwd (ortho=False):
    query [B, H, L, S], qkey / key [B, K, L, S], value [B, K, L, V] (views of [B, L, *, D] storage are read
    without copies). Returns out [B, H, L, V] (a view of [B, L, H, V] storage), out_mask [B, L],
    memory [B, K, S, V] (dtype of query) and log_norm_term [B, K, S] (fp32). All masks are derived from
    segment_idx inside the kernels."""
    assert not ortho, "orthogonal update is not supported by the CuTe kernels"
    B, H, L, S = query.shape
    K, V = value.shape[1], value.shape[3]
    C = chunk_size
    assert L % C == 0 and C % 128 == 0 and query.dtype == torch.bfloat16
    nc = L // C
    dev = query.device
    has_prev = prev_key is not None
    has_mem = memory is not None
    has_seg = segment_idx is not None
    if not has_mem:
        assert log_norm_term is None and not has_prev
    q, qk, k, v = _blhs(query), _blhs(qkey), _blhs(key), _blhs(value)
    seg = segment_idx.contiguous() if has_seg else _dummy(torch.int64, dev)
    prev_cnt = prev_segment_count.contiguous() if (has_seg and has_prev) else _dummy(torch.int64, dev)
    seg_off = C if (has_seg and has_prev) else 0  # segment ids of the current rows start after the previous chunk
    seg_stride = seg.shape[1] if has_seg else 0
    R = min(8, C // 128)

    # phase 1: update chunks (the previous call's last chunk at index 0 when given, then chunks 0 .. nc-2)
    nu_tot = nc - 1 + int(has_prev)
    part = torch.empty(B, K, max(nu_tot, 1), R, S, 2, dtype=torch.float32, device=dev)
    AB = torch.zeros(B, K, max(nu_tot, 1), S, S + V, dtype=torch.float32, device=dev)
    if nc > 1:
        _chunk_ab(k, qk, v, seg, part, AB, K, S, V, C, R, nc - 1, int(has_prev), seg_off, has_seg)
    if has_prev:
        _chunk_ab(_blhs(prev_key), _blhs(prev_qkey), _blhs(prev_value), seg, part, AB, K, S, V, C, R, 1, 0, 0, has_seg)

    # phase 2: recurrence (fp32), memory resets, chunks whose memory is still empty
    MT = torch.empty(B, K, nc, V, S, dtype=torch.bfloat16, device=dev)
    mem_out = torch.empty(B, K, S, V, dtype=query.dtype, device=dev)
    lnt_out = torch.empty(B, K, S, dtype=torch.float32, device=dev)
    dead = torch.empty(B, nc, dtype=torch.uint8, device=dev)
    rec_vt = next((vt for vt in (64, 32, 16) if V % vt == 0 and B * K * (V // vt) >= 128), 16)
    rec_key = ("rec", K, S, V, C, R, has_mem, has_prev, has_seg, rec_vt)
    f_rec = _compiled(rec_key, lambda: AwmRecurrence(K, S, V, C, R, has_mem, has_prev, has_seg, rec_vt), lambda: (
        _fake(Float32), _fake(Float32), _fake(BFloat16), _fake(Float32), _fake(Int64, 8), _fake(Int64, 8),
        _fake(_U8, 1), _fake(BFloat16), _fake(BFloat16), _fake(Float32), *([Int32(0)] * 5)))
    f_rec(part.reshape(-1), AB.reshape(-1), (memory if has_mem else mem_out).reshape(-1),
          (log_norm_term.float() if has_mem else lnt_out).reshape(-1), seg.reshape(-1), prev_cnt.reshape(-1),
          dead.reshape(-1), MT.reshape(-1), mem_out.reshape(-1), lnt_out.reshape(-1), B, nc, max(nu_tot, 1),
          seg_off, seg_stride)

    # phase 3: outputs and the output mask
    out = torch.empty(B, L, H, V, dtype=torch.bfloat16, device=dev)
    out_mask = torch.empty(B, L, dtype=torch.bool, device=dev)
    out_key = ("out", H, K, S, V, C, has_prev, has_seg)
    f_out = _compiled(out_key, lambda: AwmOutGemm(H, K, S, V, C, has_prev, has_seg), lambda: (
        _fake(BFloat16), _fake(BFloat16), _fake(BFloat16), _fake(Int64, 8), _fake(Int64, 8), _fake(_U8, 1),
        _fake(_U8, 1), *([Int32(0)] * 5)))
    vt = min(V, 256)
    num_tiles = (V // vt) * B * K * nc * (H // K) * (C // 128)
    f_out(q.reshape(-1), MT.reshape(-1), out.reshape(-1), seg.reshape(-1), prev_cnt.reshape(-1), dead.reshape(-1),
          out_mask.view(torch.uint8).reshape(-1), B, L, seg_off, seg_stride, min(_sm_count(dev), num_tiles))
    return out.transpose(1, 2), out_mask, mem_out, lnt_out
