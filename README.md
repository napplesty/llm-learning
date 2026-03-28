# LLM Learning

从零开始学习大语言模型 (LLM) 的实现原理，包含完整的代码实现和详细的注释说明。

## 项目简介

本项目通过 16 个循序渐进的模块，系统地介绍了现代大语言模型的核心组件和前沿技术。每个模块都包含：
- 详细的概念解释和可视化图示
- 完整的 PyTorch 代码实现
- 可运行的示例和测试

## 模块目录

### 基础组件 (01-05)

| 模块 | 主题 | 内容 |
|------|------|------|
| [01_tokenizer.py](01_tokenizer.py) | 分词器 | BPE (Byte Pair Encoding) 算法实现，文本到 token 的转换 |
| [02_embeddings.py](02_embeddings.py) | 词嵌入 | Token Embedding、Position Embedding、Layer Normalization |
| [03_attention.py](03_attention.py) | 注意力机制 | Scaled Dot-Product Attention、Multi-Head Attention、Causal Mask |
| [04_rope.py](04_rope.py) | 旋转位置编码 | RoPE (Rotary Position Embeddings)，相对位置感知，用于 LLaMA |
| [05_swiglu.py](05_swiglu.py) | SwiGLU 激活函数 | 门控线性单元，Swish 激活，用于现代 LLM 的 FFN |

### 架构设计 (06-08)

| 模块 | 主题 | 内容 |
|------|------|------|
| [06_moe.py](06_moe.py) | 混合专家模型 | Mixture of Experts，Top-k 路由，负载均衡，稀疏激活 |
| [07_transformer.py](07_transformer.py) | Transformer Block | 完整的 Transformer 块，Pre-Norm 架构，残差连接 |
| [08_model.py](08_model.py) | 完整 LLM | 组装所有组件，~0.1B 参数模型，文本生成 |

### 训练与优化 (09-11)

| 模块 | 主题 | 内容 |
|------|------|------|
| [09_training.py](09_training.py) | 训练流程 | 数据准备、梯度累积、Cosine 学习率调度、混合精度训练 |
| [10_advanced.py](10_advanced.py) | 高级技术 | Muon 优化器、CLIP 对比学习 |
| [11_efficiency.py](11_efficiency.py) | 效率优化 | Flash Attention、KV Cache、滑动窗口注意力、梯度检查点 |

### 微调与对齐 (12-13)

| 模块 | 主题 | 内容 |
|------|------|------|
| [12_finetuning.py](12_finetuning.py) | 参数高效微调 | LoRA、QLoRA、Prefix Tuning、Adapter Layers |
| [13_alignment.py](13_alignment.py) | 模型对齐 | RLHF、DPO、PPO、Constitutional AI |

### 前沿技术 (14-16)

| 模块 | 主题 | 内容 |
|------|------|------|
| [14_multimodal.py](14_multimodal.py) | 多模态模型 | Vision Transformer、跨模态注意力、LLaVA 架构 |
| [15_mamba_ssm.py](15_mamba_ssm.py) | Mamba/SSM | 状态空间模型，线性复杂度 O(N)，选择性状态空间 |
| [16_grpo.py](16_grpo.py) | GRPO | Group Relative Policy Optimization，DeepSeek-R1 的强化学习算法 |

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
python 01_tokenizer.py
python 03_attention.py

# 运行所有模块的测试
python run_all.py
```

## 学习路径

```
推荐学习顺序：

Tokenizer (01) → Embeddings (02) → Attention (03) → RoPE (04) → SwiGLU (05)
                                                              ↓
                                                      Transformer Block (07)
                                                              ↓
                                               MoE (06) ←→ Complete Model (08)
                                                              ↓
                                                         Training (09)
                                                              ↓
                          Efficiency (11) ←→ Advanced (10) ←→ Fine-tuning (12)
                                                              ↓
                          Multimodal (14) ←→ Alignment (13) ←→ Mamba/SSM (15)
                                                              ↓
                                                            GRPO (16)
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

### SwiGLU
```
output = (Swish(xW₁) ⊙ xW₂) W₃
其中 Swish(x) = x × sigmoid(x)
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
- [Mamba](https://arxiv.org/abs/2312.00752) - 状态空间模型
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - GRPO 算法

## License

MIT License
