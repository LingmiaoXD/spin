# SPIN模型注意力权重可视化指南

本指南介绍如何提取和可视化SPIN模型的注意力权重，以便理解模型的学习行为。

## 概述

SPIN模型使用两种注意力机制：
1. **空间图注意力（Spatial Graph Attention）**：在图的边上进行消息传递，聚合邻居节点的信息
2. **时间自注意力（Temporal Self-Attention）**：在同一节点的不同时间步之间进行注意力计算

## 快速开始

### 1. 导入可视化工具

```python
from spin.utils.visualize_attention_weights import (
    AttentionVisualizer,
    create_temporal_edge_index
)
```

### 2. 创建可视化工具实例

```python
# 假设你已经有了训练好的模型
model = SPINModel(...)  # 你的模型实例
model.load_state_dict(torch.load('checkpoint.ckpt')['state_dict'])
model.eval()

# 创建可视化工具
visualizer = AttentionVisualizer(model)
```

### 3. 准备数据

```python
# 从数据加载器获取一个样本批次
batch = next(iter(test_dataloader))

x = batch.x          # [B, T, N, C]
mask = batch.mask    # [B, T, N, C]
edge_index = batch.edge_index  # [2, E]
```

### 4. 提取注意力权重

```python
# 提取权重（这会临时修改模型的message方法）
spatial_weights, temporal_weights = visualizer.extract_weights(
    x, mask, edge_index, edge_weight=batch.edge_weight
)
```

### 5. 可视化时间注意力

```python
# 创建时间边索引（用于重构注意力矩阵）
n_time_steps = x.shape[1]  # 时间步数
temporal_edge_index = create_temporal_edge_index(
    n_time_steps=n_time_steps,
    max_distance=None,  # 或根据模型配置设置max_temporal_distance
    causal=False
)

# 可视化每一层的时间注意力
for layer_idx in temporal_weights.keys():
    visualizer.plot_temporal_attention(
        temporal_weights, 
        layer_idx=layer_idx,
        edge_index=temporal_edge_index,
        n_time_steps=n_time_steps,
        batch_idx=0,
        node_idx=0,
        save_path=f'temporal_attention_layer{layer_idx}.png'
    )
```

### 6. 可视化空间注意力统计

```python
# 可视化空间注意力的Top-K边权重
for layer_idx in spatial_weights.keys():
    visualizer.plot_spatial_attention_stats(
        spatial_weights,
        layer_idx=layer_idx,
        edge_index=edge_index,
        batch_idx=0,
        time_step=0,  # 选择要可视化的时间步
        top_k=20,     # 显示前20个最高权重的边
        save_path=f'spatial_attention_layer{layer_idx}_t0.png'
    )
```

## 完整示例

参考 `examples/visualize_attention_example.py` 获取完整的使用示例。

## 注意事项

### 1. 时间边索引的创建

时间注意力的可视化需要知道时间步之间的连接关系。`create_temporal_edge_index` 函数会创建一个全连接的时间图（除非设置了 `max_distance` 或 `causal=True`）。

如果你的模型使用了 `max_temporal_distance` 参数，应该相应设置：

```python
temporal_edge_index = create_temporal_edge_index(
    n_time_steps=n_time_steps,
    max_distance=model.encoder[0].cross_attention.max_temporal_distance,  # 从模型获取
    causal=False
)
```

### 2. 权重提取的工作原理

`AttentionVisualizer` 在提取权重时会：
1. 临时修改 `AdditiveAttention.message` 方法
2. 在执行前向传播时保存注意力权重
3. 恢复原始的 message 方法

这不会影响模型的原始功能，但建议只在评估/可视化时使用。

### 3. 权重数据的格式

提取的权重数据格式如下：

```python
# temporal_weights[layer_idx] 是一个列表，每个元素包含：
{
    'weights': torch.Tensor,  # 注意力权重 [B, T, E, 1] 或 [E, 1]
    'index': numpy.ndarray,   # 边索引（用于映射权重到具体连接）
    'size_i': int,            # 目标节点/时间步的数量
    'msg_shape': tuple        # 消息的形状
}
```

### 4. 批次和时间维度

- 可视化函数中的 `batch_idx` 参数用于选择要可视化的批次
- 对于时间注意力，权重通常在批次和时间维度上都有
- 对于空间注意力，可以指定 `time_step` 参数来选择特定时间步的权重

## 解读可视化结果

### 时间注意力热力图

- **对角线附近的高权重**：表示相邻时间步之间的强关联
- **对称性**：如果矩阵大致对称，说明模型认为时间依赖是双向的
- **局部性**：如果高权重集中在对角线附近，说明模型更关注短期依赖

### 空间注意力统计

- **Top-K边权重**：显示模型认为最重要的节点连接
- **权重分布**：可以帮助理解哪些空间关系对模型最重要
- **时间变化**：比较不同时间步的权重可以观察空间依赖的时间演化

## 常见问题

**Q: 提取权重后模型性能是否受影响？**

A: 不会。权重提取是临时的，完成后会恢复原始方法。但建议使用 `model.eval()` 模式。

**Q: 为什么时间注意力矩阵看起来不对？**

A: 可能是 `temporal_edge_index` 的创建方式不正确。检查模型的 `max_temporal_distance` 设置，或者查看模型中实际使用的时间边索引。

**Q: 如何只可视化特定节点的注意力？**

A: 目前工具会聚合所有节点的权重。如果需要节点特定的可视化，需要修改 `reconstruct_temporal_matrix` 函数。

**Q: 权重提取失败怎么办？**

A: 确保模型处于 `eval()` 模式，并且输入数据在正确的设备上（CPU或GPU）。检查模型结构是否与工具兼容。

## 高级用法

### 自定义可视化

你可以直接访问提取的权重数据进行自定义分析：

```python
# 访问原始权重数据
layer_0_temporal = temporal_weights[0]
for weight_data in layer_0_temporal:
    weights = weight_data['weights']  # 原始权重张量
    indices = weight_data['index']    # 边索引
    
    # 进行自定义分析
    # ...
```

### 批量可视化

```python
# 对多个样本进行可视化
for i, batch in enumerate(test_dataloader):
    spatial_weights, temporal_weights = visualizer.extract_weights(...)
    # 保存时使用不同的文件名
    visualizer.plot_temporal_attention(
        ..., 
        save_path=f'temporal_attention_sample{i}_layer0.png'
    )
```


