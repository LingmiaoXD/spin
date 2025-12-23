# SPIN模型参数量分析与轻量化建议

## 一、当前模型参数量估算

### 1.1 配置参数（基于 `config/imputation/spin.yaml`）

- `hidden_size`: 32
- `n_layers`: 3
- `eta`: 2
- `message_layers`: 1
- `temporal_self_attention`: True
- `input_size`: 3（avg_speed, avg_occupancy, total_vehicles）
- `n_nodes`: 假设100个节点（实际取决于数据集）

### 1.2 参数量详细分解

#### 1.2.1 位置编码器 (u_enc)
- `lin`: Linear(u_size → hidden_size) = 1 × 32 = 32
- `mlp`: MLP(32 → 32, n_layers=2) ≈ 32×32 + 32×32 = 2,048
- `node_emb`: StaticGraphEmbedding(n_nodes, hidden_size) = 100 × 32 = 3,200
- **小计**: ~5,280 参数

#### 1.2.2 输入编码器 (h_enc)
- MLP(3 → 32, n_layers=2) = 3×32 + 32×32 = 1,120
- `h_norm`: LayerNorm(32) = 64（可忽略）
- **小计**: ~1,120 参数

#### 1.2.3 嵌入层
- `valid_emb`: StaticGraphEmbedding(100, 32) = 3,200
- `mask_emb`: StaticGraphEmbedding(100, 32) = 3,200
- **小计**: 6,400 参数

#### 1.2.4 每层编码器（共3层）

每层包含：

**a) Skip Connection (x_skip)**
- Linear(3 → 32) = 3×32 + 32 = 128
- 3层总计: 384

**b) TemporalGraphAdditiveAttention (encoder)**

每层包含两个TemporalAdditiveAttention（self_attention + cross_attention）：

每个TemporalAdditiveAttention包含：
- `lin_src`: Linear(32 → 32) = 32×32 + 32 = 1,056
- `lin_tgt`: Linear(32 → 32) = 32×32 = 1,024
- `lin_skip`: Linear(32 → 32) = 32×32 = 1,024
- `msg_nn`: MLP(32 → 32, n_layers=1) = 32×32 + 32×32 = 2,048
- `msg_gate`: Linear(32 → 1) = 32 + 1 = 33
- `norm`: LayerNorm(32) = 64（可忽略）
- **每个TemporalAdditiveAttention**: ~5,185 参数

每层encoder包含：
- `self_attention`: 5,185（如果temporal_self_attention=True）
- `cross_attention`: 5,185
- `lin_skip`: 1,024
- `norm`: 64
- **每层encoder**: ~11,458 参数（如果启用self_attention）

3层总计: ~34,374 参数

**c) Readout层**
- MLP(32 → 32 → 3, n_layers=2) = 32×32 + 32×3 = 1,120
- 3层总计: 3,360

#### 1.2.5 总参数量

| 组件 | 参数量 | 占比 |
|------|--------|------|
| u_enc | ~5,280 | 10.5% |
| h_enc | ~1,120 | 2.2% |
| valid_emb + mask_emb | 6,400 | 12.7% |
| x_skip (3层) | 384 | 0.8% |
| encoder (3层) | ~34,374 | 68.3% |
| readout (3层) | 3,360 | 6.7% |
| **总计** | **~50,918** | **100%** |

**约 51K 参数（0.05M）**

> 注意：实际参数量取决于数据集中的节点数（n_nodes）。如果节点数为200，参数量约为 **~58K**；如果节点数为50，参数量约为 **~45K**。

### 1.3 轻量化配置（基于 `config/imputation/spin_lane.yaml`）

- `hidden_size`: 16（减半）
- 其他参数相同

**估算参数量**: ~15K-20K 参数（约减少60-70%）

## 二、轻量化改进建议

### 2.1 降低隐藏层维度 ⭐⭐⭐⭐⭐

**影响**: 最大，参数量与 `hidden_size²` 成正比

**建议**:
```yaml
hidden_size: 16  # 从32降到16，参数量减少约60%
# 或更激进
hidden_size: 8   # 参数量减少约85%，但可能影响性能
```

**权衡**: 
- ✅ 参数量大幅减少
- ⚠️ 可能影响模型表达能力，需要在小样本上验证性能

### 2.2 减少编码器层数 ⭐⭐⭐⭐

**影响**: 大，每层约11K参数

**建议**:
```yaml
n_layers: 2  # 从3降到2，减少约33%的编码器参数
```

**权衡**:
- ✅ 直接减少参数量
- ⚠️ 可能影响模型深度和表达能力

### 2.3 禁用时间自注意力 ⭐⭐⭐

**影响**: 中等，每层减少约5K参数

**建议**:
```yaml
temporal_self_attention: False  # 禁用时间自注意力
```

**权衡**:
- ✅ 每层减少约45%参数
- ⚠️ 可能影响时间依赖建模能力
- 💡 如果空间图注意力足够强，可以尝试

### 2.4 减少消息传递层数 ⭐⭐

**影响**: 小，每层约2K参数

**建议**:
```yaml
message_layers: 1  # 已经是1，可以保持
# 如果当前是2，可以降到1
```

### 2.5 共享嵌入层 ⭐⭐⭐

**当前**: `valid_emb` 和 `mask_emb` 是独立的（6,400参数）

**建议**: 使用单个嵌入层，通过可学习的偏移量区分：
```python
# 修改 spin/models/spin.py
self.shared_emb = StaticGraphEmbedding(n_nodes, hidden_size)  # 3,200参数
self.emb_offset = nn.Parameter(torch.zeros(hidden_size))  # 32参数
```

**节省**: ~3,200参数（约6%）

### 2.6 简化位置编码器 ⭐⭐

**当前**: PositionalEncoder包含MLP和节点嵌入

**建议**: 
- 减少MLP层数：`n_layers=1` 而不是2
- 或使用更简单的编码方式

**节省**: ~1,000参数

### 2.7 限制时间注意力范围 ⭐⭐

**当前**: `max_temporal_distance: 5` 或 `None`（全连接）

**影响**: 主要影响计算量，对参数量影响较小

**建议**:
```yaml
max_temporal_distance: 3  # 限制时间步连接范围
```

**权衡**:
- ✅ 减少计算量，提高训练速度
- ⚠️ 对参数量影响较小（主要是计算优化）

### 2.8 使用更小的Readout层 ⭐

**当前**: MLP(32 → 32 → 3, n_layers=2)

**建议**: 直接线性映射或单层MLP
```python
# 单层MLP
readout = MLP(hidden_size, output_size, n_layers=1)
```

**节省**: 每层约1,000参数，3层共3,000参数

### 2.9 知识蒸馏 ⭐⭐⭐⭐

**策略**: 
1. 训练一个较大的教师模型（当前配置）
2. 训练一个轻量化的学生模型（使用上述轻量化配置）
3. 使用教师模型的输出作为软标签训练学生模型

**优势**: 
- 保持轻量化模型的小参数量
- 通过知识蒸馏获得更好的性能

### 2.10 渐进式训练策略 ⭐⭐⭐

**策略**:
1. 先用轻量化配置训练基础模型
2. 如果数据量增加，再逐步增加模型容量

**配置示例**:
```yaml
# 阶段1：极小模型（小样本）
hidden_size: 8
n_layers: 2
temporal_self_attention: False

# 阶段2：小模型（中等样本）
hidden_size: 16
n_layers: 2
temporal_self_attention: True

# 阶段3：标准模型（大样本）
hidden_size: 32
n_layers: 3
temporal_self_attention: True
```

## 三、推荐的轻量化配置

### 3.1 轻度轻量化（保持较好性能）

```yaml
hidden_size: 16        # 从32降到16
n_layers: 2            # 从3降到2
eta: 1                 # 从2降到1
message_layers: 1      # 保持
temporal_self_attention: True  # 保持
max_temporal_distance: 3  # 限制时间范围
```

**预期参数量**: ~15K-20K（减少60-70%）
**预期性能**: 可能下降5-10%

### 3.2 中度轻量化（平衡性能与参数量）

```yaml
hidden_size: 12        # 进一步降低
n_layers: 2            # 保持2层
eta: 1                 # 保持
message_layers: 1      # 保持
temporal_self_attention: False  # 禁用时间自注意力
max_temporal_distance: 3
```

**预期参数量**: ~10K-12K（减少75-80%）
**预期性能**: 可能下降10-15%

### 3.3 极度轻量化（最小参数量）

```yaml
hidden_size: 8         # 最小隐藏层
n_layers: 2            # 保持2层
eta: 1                 # 保持
message_layers: 1      # 保持
temporal_self_attention: False  # 禁用
max_temporal_distance: 2  # 更小的时间范围
```

**预期参数量**: ~5K-8K（减少85-90%）
**预期性能**: 可能下降15-25%，需要仔细调优

## 四、小样本数据集优化策略

### 4.1 数据增强

1. **时间窗口滑动**: 增加训练样本数量
2. **图结构增强**: 使用不同的图连接策略
3. **噪声注入**: 在训练时添加轻微噪声提高鲁棒性

### 4.2 正则化策略

```yaml
l2_reg: 0.001  # 增加L2正则化，防止过拟合
dropout: 0.1   # 在模型中添加dropout（需要修改代码）
```

### 4.3 早停策略

```yaml
patience: 10   # 更早停止，防止过拟合
min_delta: 0.001
```

### 4.4 学习率调整

```yaml
lr: 0.0005     # 降低学习率，更稳定的训练
lr_scheduler: magic
```

### 4.5 批次大小调整

```yaml
batch_size: 2  # 减小批次大小，增加梯度更新频率
batches_epoch: 500  # 增加每epoch的批次数
```

## 五、实施建议

### 优先级排序

1. **高优先级**（立即实施）:
   - 降低 `hidden_size` 到 16
   - 减少 `n_layers` 到 2
   - 限制 `max_temporal_distance`

2. **中优先级**（根据性能调整）:
   - 禁用 `temporal_self_attention`（如果性能可接受）
   - 简化 readout 层
   - 共享嵌入层

3. **低优先级**（进一步优化）:
   - 简化位置编码器
   - 知识蒸馏
   - 渐进式训练

### 验证步骤

1. 使用轻量化配置训练模型
2. 在验证集上评估性能
3. 如果性能下降过多，逐步恢复某些配置
4. 找到性能与参数量的最佳平衡点

## 六、总结

**当前参数量**: 约 **50K-60K** 参数（取决于节点数）

**轻量化后**: 可降至 **10K-20K** 参数（减少60-80%）

**关键优化点**:
1. `hidden_size`: 最大影响（与参数量平方相关）
2. `n_layers`: 直接影响层数
3. `temporal_self_attention`: 每层减少约45%参数

**小样本优化**: 结合轻量化模型 + 数据增强 + 正则化 + 早停策略
