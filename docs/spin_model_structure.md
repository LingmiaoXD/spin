# SPIN模型结构详解

## 模型架构概述

SPIN (Spatiotemporal Imputation Network) 是一个基于图注意力机制的时空数据插补模型，专门设计用于处理高度稀疏的时空图数据。

## 核心组件

### 1. 位置编码层 (Positional Encoding)

**位置**: `spin/layers/postional_encoding.py`

- **功能**: 为每个(节点, 时间步)对生成时空位置编码
- **组成**:
  - 节点嵌入 (Node Embedding): 使用`StaticGraphEmbedding`为每个节点学习静态表示
  - 时间位置编码: 使用`PositionalEncoding`为时间步添加位置信息
  - MLP编码器: 将节点嵌入和时间编码融合

**输入**: 
- `u`: 时间特征 `[batch, steps, channels]`
- `node_index`: 节点索引（可选）

**输出**: 
- `q`: 位置编码 `[batch, steps, nodes, hidden_size]`

### 2. 输入编码层

**位置**: `spin/models/spin.py` (第34-40行)

```python
self.h_enc = MLP(input_size, hidden_size, n_layers=2)  # 值编码
self.h_norm = LayerNorm(hidden_size)  # 归一化
```

- **功能**: 将输入数据编码为隐藏表示
- **处理**: 
  - 缺失值被置零（whiten）
  - 与位置编码`q`相加: `h = h_enc(x_masked) + q`
  - 缺失位置的`h`被替换为`q`（确保不传播缺失值）

### 3. 多层编码器 (Multi-Layer Encoder)

**位置**: `spin/models/spin.py` (第45-66行)

每层包含：

#### 3.1 时空图注意力层 (TemporalGraphAdditiveAttention)

**位置**: `spin/layers/temporal_graph_additive_attention.py`

这是SPIN的核心层，同时进行两种注意力：

1. **空间图注意力 (Cross-Attention)**:
   - 在图的边（edge_index）上进行消息传递
   - 聚合邻居节点的信息
   - 使用`mask_spatial`参数控制是否mask空间注意力

2. **时间自注意力 (Self-Attention)**:
   - 在同一节点的不同时间步之间进行注意力
   - 使用`mask_temporal`参数控制是否mask时间注意力
   - 可选：通过`temporal_self_attention`参数启用/禁用

**注意力机制**:
- 使用**加性注意力 (Additive Attention)**，而非点积注意力
- 通过`msg_gate`生成注意力权重
- 支持`softmax`或`l1`归一化（`reweight`参数）

#### 3.2 Skip Connection

```python
skip_connection = self.x_skip[l](x_masked) * mask_expanded
h = h + skip_connection.clone().detach()
```

- 从原始输入（仅有效值）添加跳跃连接
- 帮助梯度流动和特征保留

#### 3.3 区分有效值和缺失值

在第`eta`层之后（`l == self.eta`）：

```python
valid = self.valid_emb(token_index=node_index)  # 有效值嵌入
masked = self.mask_emb(token_index=node_index)   # 缺失值嵌入
h = torch.where(mask_expanded.bool(), h + valid, h + masked)
```

- 使用不同的嵌入来区分有效观测和缺失值
- 让模型学习不同的表示模式

#### 3.4 Readout层

```python
target_readout = self.readout[l](h[..., target_nodes, :])
```

- 每层都输出一个插补结果
- 最终使用最后一层的输出作为最终插补

## 前向传播流程

```
输入 x [B, T, N, C] + mask [B, T, N, C] + 图结构 edge_index
    ↓
1. 位置编码: q = u_enc(u)  [B, T, N, H]
    ↓
2. 值编码: h = h_enc(x_masked) + q
    ↓
3. 替换缺失位置: h = where(mask, h, q)
    ↓
4. 归一化: h = h_norm(h)
    ↓
5. 多层编码 (n_layers次):
   for l in range(n_layers):
       if l == eta:
           h = where(mask, h + valid_emb, h + mask_emb)
       skip = x_skip[l](x_masked) * mask
       h = h + skip
       h = encoder[l](h, edge_index, mask)  # 时空图注意力
       imputation[l] = readout[l](h)
    ↓
输出: x_hat (最后一层插补) + 中间层插补列表
```

## 关键设计特点

1. **稀疏注意力**: 只关注有效观测，不传播缺失值
2. **时空联合建模**: 同时考虑空间（图）和时间（序列）依赖
3. **渐进式插补**: 每层都产生插补结果，可以用于辅助训练
4. **位置感知**: 通过位置编码让模型理解时空结构

## 模型参数

- `hidden_size`: 隐藏层维度
- `n_layers`: 编码器层数（默认3-4层）
- `eta`: 从哪一层开始区分有效值和缺失值（默认2-3）
- `message_layers`: 消息传递的MLP层数
- `temporal_self_attention`: 是否启用时间自注意力
- `reweight`: 注意力权重归一化方式（'softmax'或None）

## 与Hierarchical版本的差异

`SPINHierarchicalModel`增加了：
- 层次化表示`z`：节点级别的全局表示
- 更复杂的注意力机制：同时更新`h`和`z`
- 更多的嵌入层来区分不同状态

















