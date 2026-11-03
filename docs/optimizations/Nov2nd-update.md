# 2026-11-02 update: what was done on top of the original base

Base: `main` f5f8687. Branch: `gekko-optimizations`. Per-architecture reports: [B200.md](B200.md),
[H100.md](H100.md), [H200.md](H200.md); one-page summary [SUMMARY.md](SUMMARY.md); full kernel table
[KERNELS.md](KERNELS.md).

在 `main`（f5f8687）之上，`gekko-optimizations` 分支主要做了五类事情：

## 1. 用 CuTe DSL 重写了 Gekko 的核心算子

B200 用 tcgen05 / TMEM，H100 / H200 用 wgmma，运行时按架构选路径，都有对 fp32 / fp64 参考的单测。

- Timestep decay norm 前向 / 反向：替换 CUB，每次调用 15-20x，接近 HBM 带宽上限。
- Adaptive working memory 前向 / 反向：替换 torch / cuBLAS，前向 6-11x，反向 9-16x。
- Sliding chunk attention（B200）：替换 swift 的物化注意力矩阵，0.9B 单步 1151 → 590 ms，峰值显存 139 → 40 GB；
  V = 2048 的 7.1B 反向走物化 P / dS + GEMM 路径，可选 MXFP8 dP。Hopper 上保留 xattn / swift，但给 xattn 打了补丁、
  融合了 swift 的反向。
- 一批融合的逐元素 / 归一化核：RoPE、q/k prep（rmsnorm + rope + softmax）、conv + q/k prep 融合、attention gate 链、
  SwiGLU、group layer norm、cross-entropy 行核、causal conv 前向。0.9B 这部分步时 527 → 315 ms。

## 2. FP8 GEMM

- B200：MXFP8（e4m3 + e8m0 / 32，cuBLASLt），量化融合进上游核，GEMM 时间降 1.75-1.84x，步长 -8% 到 -27%。
- Hopper：blockwise e4m3（1 x 128 / 128 x 128），残差加在 epilogue 里；可选 DeepGEMM 后端（H200 上每个 GEMM 再快
  1.10-1.35x）；fp8 all-gather 省显存。

## 3. 训练步与 FSDP2 层面的改动

- 残差加融合进 GEMM epilogue，`recompute_conv_qk=False` 保存前向激活，`fsdp_layers_per_group` 分组 all-gather，
  `reshard_after_forward=False`。
- reduce-scatter copy-in 改为 Triton 核、梯度裁剪直接在 flat buffer 上做、本地分片的 fused AdamW、Hopper 上的 NCCL
  对称内存集合通信、`balance_dp_rows` 按注意力代价均衡各 rank 的行。
- 融合 block 的 CUDA graph 回放（4B 用来消除步长双峰；H200 上遇 EOS 打包数据还不能用）。
- 顺手修了几个 bug：`create_on_gpu` 残留的 TorchFunctionMode（4B 单卡 554 → 444 ms）、AWM 单 chunk 时的崩溃 / 挂起、
  conv_qk_prep 的 NaN、sm_100 上 `_mul2` / `_fma2` 递归。

## 4. 端到端结果（8 卡，每步 fwd + bwd 中位数）

| 模型 | B200 | H100 | H200 |
|---|---|---|---|
| 0.9B B2 L16384 | 1258 → **270** ms (**4.7x**) | 865 → **465** (**1.86x**) | 1075 → **444** (**2.42x**) |
| 4B B1 L8192 | 967 → **250** (**3.9x**) | 667 → **393-405** (**1.68x**) | 762 → **419** (**1.82x**) |
| 7.1B（B200 L16384；Hopper L8192） | 1150 → **550** (**2.1x**) | 1003 → **584** (**1.71x**) | 1082 → **536** (**2.02x**) |

数值上：bf16 项与原实现同批次 loss 差 ≤ 2e-4，fp8 约 1e-3，各架构都做了 100-200 步 soak。

## 5. 文档与工程整理

- `docs/optimizations/`：README（配置项、环境开关、测试）、B200 / H100 / H200 三份报告、SUMMARY（一页）、KERNELS
  （完整 kernel 表）、xattn 补丁、复现用 harness。
- 原来的 5 + 2 个开发 / hand-off 分支合并成这一个分支，旧历史在本机 bundle 里备份。

## 待决定

主要是默认值：fp8、对称集合通信、DeepGEMM 是否默认打开；xattn 补丁是否上游；H200 host-bound 的行在裸机上重测。
