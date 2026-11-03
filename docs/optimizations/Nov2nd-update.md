# Gekko 训练优化：2026-11-02 进展汇报

基线：`main` f5f8687。分支：`gekko-optimizations`（B200 / H100 / H200 的优化都在这一个分支，运行时按架构选路径）。
详细报告：[B200.md](B200.md)、[H100.md](H100.md)、[H200.md](H200.md)；英文一页版 [SUMMARY.md](SUMMARY.md)；完整
kernel 表 [KERNELS.md](KERNELS.md)；配置项、环境开关、测试清单见 [README.md](README.md)。

## 1. 结论

- 三个架构上的训练步都显著加快：**B200 2.1-4.7x，H100 1.7-1.9x，H200 1.8-2.4x**（8 卡 FSDP2，同一模型配置）。
- 数值与原实现一致：bf16 项同批次 loss 差 ≤ 2e-4，fp8 约 1e-3，各架构 100-200 步 soak 通过。
- 所有新路径都是可配置的（`ModelConf` 字段 / 环境变量），默认值未改；fp8、DeepGEMM 等是否默认打开待定（见第 6 节）。
- 代码 + 报告 + 复现脚本都在一个分支上；原来的 7 个开发 / hand-off 分支已合并删除。

## 2. 端到端结果（8 卡，每步 fwd + bwd 中位数，ms）

| 模型 | B200 | H100 | H200 |
|---|---|---|---|
| gekko-0.9B B2 L16384 | 1258 → **270**（**4.7x**） | 865 → **465**（**1.86x**） | 1075 → **444**（**2.42x**） |
| gekko-4B B1 L8192 | 967 → **250**（**3.9x**） | 667 → **393-405**（**1.68x**） | 762 → **419**（**1.82x**） |
| gekko-7.1B（B200 L16384；Hopper L8192） | 1150 → **550**（**2.1x**） | 1003 → **584**（**1.71x**） | 1082 → **536**（**2.02x**） |
| 最佳配置 tokens / s / GPU | 121k / 33k / 30k | 70k / 20k / 14k | 74k / 20k / 15k |
| 最佳配置峰值显存 GiB | 73 / 82 / 134 | 69 / 55 / 68 | 69 / 55 / 86 |

- 起点 = CUB TSDN、torch AWM、swift / 上游 xattn 注意力、torch 逐元素算子、bf16 GEMM（B200 的起点在单卡上测，每卡
  batch 相同，8 卡只会更慢；其余均为 8 卡）。
- 最佳 = 全部 kernel 打开 + `recompute_conv_qk=False` + `fsdp_layers_per_group` > 1 + `reshard_after_forward=False`
  + `fp8_gemm=True`，再加各架构的附加项：B200 4B 的 CUDA graph、B200 7.1B 的 `fp8_attn_dp`、Hopper 的对称内存集合
  通信与 `balance_dp_rows`、H200 7.1B 的 DeepGEMM。纯 bf16（不用 fp8）的 B200 为 308 / 270 / 771 ms。
- H200 7.1B 在 L16384 下不做 FFN 重计算可以放下（119 GiB）：1048 ms，15.4k tokens / s / GPU（H100 OOM）。

## 3. 做了什么（按类别）

| 类别 | 做了什么 | 替换 / 改动的对象 | 效果 |
|---|---|---|---|
| <sub>CuTe 算子：Timestep decay norm</sub> | <sub>前向 / 反向 look-back scan 核（B200 tcgen05 免费路径，Hopper 标量路径）</sub> | <sub>CUB scan 核</sub> | <sub>每次调用 15-20x，接近 HBM 上限；0.9B 每步 120 → 2.3 ms</sub> |
| <sub>CuTe 算子：Adaptive working memory</sub> | <sub>chunk GEMM + 递推 + 输出 GEMM，前向 / 反向（B200 tcgen05 / TMEM，Hopper wgmma）</sub> | <sub>torch / cuBLAS 实现</sub> | <sub>前向 6-11x，反向 9-16x（0.9B 反向每步 99 → 6 ms）</sub> |
| <sub>CuTe 算子：Sliding chunk attention（B200）</sub> | <sub>flash 风格前向 / 反向，TMEM 操作数；V = 2048 的 7.1B 反向走物化 P / dS + GEMM，可选 MXFP8 dP</sub> | <sub>swift（物化注意力矩阵 + cuBLAS）</sub> | <sub>0.9B 单步 1151 → 590 ms，峰值显存 139 → 40 GB；7.1B 反向每层 2.2x；前向达 tensor pipe 峰值 ~95%</sub> |
| <sub>Hopper 注意力</sub> | <sub>xattn 1.1.1 本地补丁（hdim 256 两段反向、GQA K/V 复用、预计算 dO·O）、swift 反向 dS 融合</sub> | <sub>上游 xattn / swift</sub> | <sub>xattn 反向 1.1-1.2x，swift 7.1B 每层 1.75x</sub> |
| <sub>融合逐元素 / 归一化核</sub> | <sub>RoPE、q/k prep（rmsnorm + rope + softmax）、conv + q/k prep 融合、attention gate 链、SwiGLU、group LN、cross-entropy 行核、causal conv 前向</sub> | <sub>一串 torch 逐元素算子</sub> | <sub>0.9B 这部分步时 527 → 315 ms；H100 上 RoPE 6x、block 逐元素 3.5x、conv + q/k prep 1.7x / 2.5x</sub> |
| <sub>FP8 GEMM（B200）</sub> | <sub>MXFP8（e4m3 + e8m0 / 32，cuBLASLt），量化融合进上游核，权重每步量化一次</sub> | <sub>bf16 cuBLAS</sub> | <sub>GEMM 时间 1.75-1.84x 降低；步长 -8% 到 -27%</sub> |
| <sub>FP8 GEMM（Hopper）</sub> | <sub>blockwise e4m3（1 x 128 / 128 x 128），残差加在 epilogue；可选 DeepGEMM 后端；fp8 all-gather</sub> | <sub>bf16 cuBLAS</sub> | <sub>每个 GEMM 1.5-1.7x；H200 上 DeepGEMM 再 1.10-1.35x；fp8 all-gather 省 2.6-6 GiB</sub> |
| <sub>训练步结构</sub> | <sub>残差加融合进 GEMM epilogue；`recompute_conv_qk=False` 保存前向激活；TSDN 步数前向传反向</sub> | <sub>融合 block 的重计算与多余 pass</sub> | <sub>单卡 0.9B 322 → 305，4B 288 → 273 ms</sub> |
| <sub>FSDP2 / 通信</sub> | <sub>`fsdp_layers_per_group` 分组 all-gather、`reshard_after_forward=False`、Triton reduce-scatter copy-in、梯度除法折进 copy-in、Hopper 对称内存集合通信、`balance_dp_rows`</sub> | <sub>FSDP2 默认行为、`torch._chunk_cat`</sub> | <sub>8 卡 0.9B 348 → 326、7.1B 828 → 818 ms；H200 上分组是最大收益项</sub> |
| <sub>优化器 / 裁剪</sub> | <sub>在 flat 梯度 buffer 上做梯度裁剪、本地分片上的 fused AdamW</sub> | <sub>DTensor 分发的 clip 与 AdamW</sub> | <sub>8 卡 clip 9 → 4 ms，optim 13 → 3 ms</sub> |
| <sub>CUDA graph</sub> | <sub>融合 block 的前向 / 反向按层捕获回放（`cuda_graph_block`）</sub> | <sub>eager 发射</sub> | <sub>B200 4B 消除步长双峰（251 / 280 → 251）；H200 无 EOS 时 4B -6%，有 EOS 打包数据还不能用</sub> |
| <sub>Bug 修复</sub> | <sub>`create_on_gpu` 残留的 TorchFunctionMode；AWM 单 chunk 崩溃 / 挂起；conv_qk_prep NaN；sm_100 `_mul2` / `_fma2` 递归</sub> | - | <sub>4B 单卡 554 → 444 ms（第一项）</sub> |
| <sub>测试</sub> | <sub>每个 kernel 对 fp32 / fp64 参考的单测（130+），fp8 block 级测试，端到端 parity 与 100-200 步 soak</sub> | - | <sub>bf16 项 loss 差 ≤ 2e-4，fp8 约 1e-3</sub> |
| <sub>文档 / 工程整理</sub> | <sub>`docs/optimizations/`：README、三份架构报告、SUMMARY、KERNELS、本文件、xattn 补丁、复现 harness；7 个开发 / hand-off 分支合成一个分支</sub> | - | <sub>一个分支，三份架构报告</sub> |

## 4. 各架构要点

- **B200**：收益最大的是 CuTe SCA（去掉物化注意力矩阵）和 MXFP8；之后步长已是 GPU-bound，8 卡运行在 1000 W 功耗上限
  （GEMM 阶段降频），所以 CUDA graph 对 4B 只带来稳定性、对 0.9B / 7.1B 无增益。
- **H100**：bf16 阶段（kernel 移植、融合、FSDP / NCCL 配置）贡献约 55-70% 的节省，fp8 贡献其余；bf16 步长也撞功耗上限。
- **H200**：同一份 Hopper 代码直接运行；访存受限核快 1.26-1.44x，GEMM 几乎不变。测试节点是 KVM 虚拟机、主机时钟双档，
  步长 host-bound，因此分组 all-gather 等减少主机工作的开关收益最大，这些行需要在裸机上重测。141 GB 让 7.1B L16384
  不用重计算；swift SCA 在 0.9B / 4B 上仍 OOM。

## 5. 数值验证

- 每个 kernel 有对 fp32 / fp64 参考的单测；fp8 有 block 级测试（小模型 fp8 vs bf16 的 loss、梯度）。
- 端到端同批次全开 vs 全关：bf16 项 loss 差 ≤ 2e-4，梯度范数差 ≤ 1e-3。
- fp8 soak：B200 0.9B 100 步均值差 +0.0009；H100 7.1B 150 步、4B / 0.9B 200 步；H200 三个预设各 150 步，结尾 10 步均值
  与 bf16 一致（7.1B 10.3960 vs 10.3933）。DeepGEMM 的 soak 在 120 步后有更高的后期尖峰，所以仍是 opt-in。

## 6. 待决定

1. 默认值：`fp8_gemm`、Hopper 的 `XLLM_FSDP_SYMM=1`、DeepGEMM（H200 7.1B -2~-3%）是否默认打开。
2. xattn 补丁是否推到上游（H100 / H200 的注意力数字依赖它们）。
3. CUDA graph 配合 EOS 打包数据需要 xattn 改成设备侧 / 定长 varlen 元数据（H200 4B 约 -6%），是否投入。
4. 训练 recipe：H200 上更大的每卡 batch 可以摊薄主机开销；7.1B L16384 不重计算现已可行。
5. H200 host-bound 的行在裸机 H200 上重测。

## 7. 下一步候选

- B200：功耗上限下的能效（更少的访存 pass）、7.1B 注意力反向的 dense GEMM 子块化（约 25% 的 dP / dV 时间）。
- Hopper：xattn varlen 元数据改造（为 CUDA graph）、DeepGEMM 真实数据 loss 对比后决定默认值。
- 通用：把推荐配置写进训练配置的预设，减少开关数量。
