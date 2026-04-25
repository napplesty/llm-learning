# LLM Learning

从零开始学习大语言模型 (LLM) 的实现原理，包含完整的代码实现和详细的注释说明。

## 项目简介

本项目通过多个主题模块，系统地介绍了现代大语言模型的核心组件和前沿技术。每个模块都包含：
- 详细的概念解释和可视化图示
- 完整的 PyTorch 代码实现
- 可运行的示例和测试

## 模块目录

### fundamentals — 基础组件

文本处理、表示学习与注意力机制的基础构建块。

| 文件 | 主题 | 内容 |
|------|------|------|
| [fundamentals/tokenizer.py](fundamentals/tokenizer.py) | 分词器 | BPE (Byte Pair Encoding) 算法实现，文本到 token 的转换 |
| [fundamentals/embeddings.py](fundamentals/embeddings.py) | 词嵌入 | Token Embedding、Position Embedding、Layer Normalization |
| [fundamentals/attention.py](fundamentals/attention.py) | 注意力机制 | Scaled Dot-Product Attention、Multi-Head Attention、Causal Mask |

### position_and_activation — 位置编码与激活函数

让模型感知序列位置，以及现代 LLM 使用的门控激活函数。

| 文件 | 主题 | 内容 |
|------|------|------|
| [position_and_activation/rope.py](position_and_activation/rope.py) | 位置编码 | RoPE、p-RoPE (Gemma 4)、ALiBi (Attention with Linear Biases) |
| [position_and_activation/swiglu.py](position_and_activation/swiglu.py) | 门控激活 | SwiGLU (LLaMA)、GeGLU (Gemma)、GLU 变体对比 |

### architecture — 架构设计

Transformer 块的组装、稀疏扩展、残差连接改进与条件记忆。

| 文件 | 主题 | 内容 |
|------|------|------|
| [architecture/transformer_block.py](architecture/transformer_block.py) | Transformer Block | 完整的 Transformer 块，Pre-Norm 架构，残差连接 |
| [architecture/mixture_of_experts.py](architecture/mixture_of_experts.py) | 混合专家模型 | Mixture of Experts，Top-k 路由，负载均衡，稀疏激活 |
| [architecture/complete_model.py](architecture/complete_model.py) | 完整 LLM | 组装所有组件，~0.1B 参数模型，文本生成 |
| [architecture/hyper_connections.py](architecture/hyper_connections.py) | 超连接 | DeepSeek mHC，Sinkhorn-Knopp 双随机约束，残差连接升级 |
| [architecture/engram.py](architecture/engram.py) | 条件记忆 | DeepSeek Engram，N-gram 哈希查找，上下文门控融合 |

### training — 训练与优化

模型训练流程、高级优化器与内存优化技术。

| 文件 | 主题 | 内容 |
|------|------|------|
| [training/training_loop.py](training/training_loop.py) | 训练流程 | 数据准备、梯度累积、Cosine 学习率调度、混合精度训练 |
| [training/optimizers_and_checkpoint.py](training/optimizers_and_checkpoint.py) | 优化器与内存 | Muon 优化器、梯度检查点 |

### inference — 推理优化

提升推理速度与降低内存占用的关键技术。

| 文件 | 主题 | 内容 |
|------|------|------|
| [inference/inference_optimization.py](inference/inference_optimization.py) | 推理效率 | Flash Attention、KV Cache、滑动窗口注意力 |
| [inference/mla.py](inference/mla.py) | 潜在注意力 | DeepSeek MLA，KV Cache 低秩压缩，10-20x 内存降低 |

### adaptation — 模型适配

在下游任务上高效微调大模型的参数高效方法。

| 文件 | 主题 | 内容 |
|------|------|------|
| [adaptation/parameter_efficient_finetuning.py](adaptation/parameter_efficient_finetuning.py) | 参数高效微调 | LoRA、QLoRA、Prefix Tuning、Adapter Layers |

### alignment — 模型对齐

让模型行为符合人类价值观的技术，从 RLHF 到最新的 GRPO。

| 文件 | 主题 | 内容 |
|------|------|------|
| [alignment/alignment.py](alignment/alignment.py) | 传统对齐 | RLHF、DPO、PPO、Constitutional AI |
| [alignment/grpo.py](alignment/grpo.py) | GRPO | Group Relative Policy Optimization，DeepSeek-R1 的强化学习算法 |

### beyond_text — 超越文本

多模态理解与 Transformer 替代架构。

| 文件 | 主题 | 内容 |
|------|------|------|
| [beyond_text/multimodal.py](beyond_text/multimodal.py) | 多模态模型 | Vision Transformer、SigLIP、CLIP 对比学习、跨模态注意力、LLaVA 架构 |
| [beyond_text/mamba_ssm.py](beyond_text/mamba_ssm.py) | Mamba/SSM | 状态空间模型，线性复杂度 O(N)，选择性状态空间 |

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 运行单个模块
python fundamentals/tokenizer.py
python fundamentals/attention.py

# 运行所有模块的测试
python run_all.py
```

## 学习路径

```
推荐学习顺序：

Tokenizer ──► Embeddings ──► Attention
                              │
                              ▼
                    Position Encodings (RoPE / p-RoPE / ALiBi)
                              │
                              ▼
                    Gated Activations (SwiGLU / GeGLU)
                              │
                              ▼
                      Transformer Block
                              │
                              ▼
              ┌───────────────┴───────────────┐
              ▼                               ▼
      Mixture of Experts              Complete Model
              │                               │
              ▼                               ▼
      mHC (超连接)                    Training Pipeline
      Engram (条件记忆)                       │
                                              ▼
              ┌───────────────┬───────────────┴───────────────┐
              ▼               ▼                               ▼
      Advanced Optimizers   Inference Optimization    Fine-tuning (LoRA)
      (Muon / Checkpoint)   (FlashAttn / KV Cache                 │
                            / Sliding Window / MLA)               ▼
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                       Model Alignment                    Multimodal
                                       (RLHF / DPO / GRPO)               (ViT / SigLIP / CLIP)
                                                                              │
                                                                              ▼
                                                                        Mamba / SSM
```

## 核心概念速查

### 注意力机制
```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

### RoPE 旋转编码
```
位置 m 的向量旋转角度 = m × θ
Q_m · K_n ∝ cos((m-n) × θ)  # 相对位置
```

### p-RoPE (Gemma 4)
```
仅对前 p 比例维度对施加旋转 (p=0.25)
剩余维度保持语义不变，避免长距离错位
```

### SwiGLU / GeGLU
```
SwiGLU: output = (Swish(xW_gate) ⊙ xW_up) W_down    # LLaMA
GeGLU:  output = (GELU-Tanh(xW_gate) ⊙ xW_down) W_up # Gemma
```

### MLA (DeepSeek)
```
c^KV = W_CKV · h          # 压缩到潜空间 (d_c << d_model)
K, V = W_DK·c^KV, W_DV·c^KV  # 推理时只缓存 c^KV
Cache reduction: (num_heads × d_k) / d_c  ≈ 10-20x
```

### mHC (DeepSeek)
```
H_res → Sinkhorn-Knopp → 双随机矩阵
保证: 行和=1, 列和=1 → 加权平均 → 恢复恒等映射
```

### Engram (DeepSeek)
```
N-gram → 多头哈希 → 查表 → 门控注入
 gate = σ(⟨h_t, W_K · e_t⟩ / √d)
 h'_t = h_t + gate ⊙ (W_V · e_t)
```

### LoRA
```
y = Wx + BAx
原始: d² 参数
LoRA: 2×d×r 参数 (r << d)
```

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE 位置编码
- [LLaMA](https://arxiv.org/abs/2302.13971) - LLaMA 架构
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU / GeGLU
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - MLA 多潜在注意力
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) - 超连接残差
- [Engram](https://arxiv.org/abs/2601.07372) - 条件记忆
- [Mamba](https://arxiv.org/abs/2312.00752) - 状态空间模型
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - GRPO 算法

## License

MIT License
