# 车道交互规则实现指南

## 概述

车道交互规则用于控制不同车道之间的图连接，模拟真实道路中的车道标线规则：
- **虚线 (dashed)**：允许车道间交互，建立图连接
- **实线 (solid)**：禁止车道间交互，不建立图连接

## 数据格式

### 输入数据格式

在原始数据中添加 `lane_interaction` 列来标识车道交互规则：

```csv
timestamp,lane_id,spatial_id,speed,spacing,lane_interaction
2024-01-01 00:00:00,lane_0,lane_0_0000,30.5,25.2,dashed
2024-01-01 00:00:00,lane_1,lane_1_0000,32.1,23.8,dashed
2024-01-01 00:00:00,lane_2,lane_2_0000,28.9,26.5,solid
2024-01-01 00:00:10,lane_0,lane_0_0001,31.2,24.8,solid
2024-01-01 00:00:10,lane_1,lane_1_0001,33.5,22.1,solid
2024-01-01 00:00:10,lane_2,lane_2_0001,29.8,25.9,solid
```

### 交互规则类型

| 类型 | 描述 | 图连接 | 说明 |
|------|------|--------|------|
| `dashed` | 虚线 | ✅ 建立连接 | 允许车道间交互 |
| `solid` | 实线 | ❌ 不建立连接 | 禁止车道间交互 |
| `interact` | 可交互 | ✅ 建立连接 | 等同于dashed |
| `no_interact` | 不可交互 | ❌ 不建立连接 | 等同于solid |

## 实现原理

### 1. 图连接构建流程

```python
def _build_graph_connectivity(self):
    # 1. 车道内连接（纵向）
    for lane_id in self.lane_ids:
        # 连接同一车道内的相邻节点
        connect_lane_internal_nodes(lane_id)
    
    # 2. 车道间连接（横向，基于交互规则）
    for lane1, lane2 in lane_pairs:
        for space1 in lane1_spaces:
            for space2 in lane2_spaces:
                if can_lanes_interact(space1, space2):
                    # 建立跨车道连接
                    add_cross_lane_connection(space1, space2)
```

### 2. 交互规则判断

```python
def _can_lanes_interact(self, space1, space2, spatial_to_interaction):
    # 获取两个空间位置的交互规则
    interaction1 = spatial_to_interaction[space1]
    interaction2 = spatial_to_interaction[space2]
    
    # 检查是否都允许交互
    can_interact1 = check_interaction_rule(interaction1)
    can_interact2 = check_interaction_rule(interaction2)
    
    # 两个位置都允许交互才能建立连接
    return can_interact1 and can_interact2
```

## 使用方法

### 1. 基本使用

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset

# 加载包含交互规则的数据
dataset = LaneTrafficDataset(
    data_path="lane_data_with_interaction.csv",
    lane_interaction_col='lane_interaction'
)

# 获取图连接矩阵
adj = dataset.get_connectivity()
```

### 2. 创建交互规则

```python
from spin.datasets.lane_data_utils import LaneDataProcessor

processor = LaneDataProcessor()

# 定义交互区域
interaction_zones = [
    {
        'lanes': ['lane_0', 'lane_1'],
        'start_pos': 0,
        'end_pos': 200,
        'type': 'dashed'  # 0-200米可以交互
    },
    {
        'lanes': ['lane_0', 'lane_1'],
        'start_pos': 200,
        'end_pos': 400,
        'type': 'solid'   # 200-400米不可以交互
    }
]

# 创建交互规则
lane_rules = processor.create_lane_interaction_rules(
    lane_ids=['lane_0', 'lane_1', 'lane_2'],
    interaction_zones=interaction_zones
)
```

### 3. 处理原始数据

```python
# 处理包含交互规则的原始数据
processed_data = processor.process_raw_data(
    raw_data=your_raw_data,
    lane_info=lane_rules
)
```

## 高级配置

### 1. 自定义交互规则

```python
def custom_interaction_rule(lane_id, spatial_pos, context):
    """自定义交互规则函数"""
    # 基于车道ID和位置的复杂规则
    if lane_id == 'lane_0' and 100 <= spatial_pos <= 300:
        return 'dashed'
    elif lane_id == 'lane_1' and 200 <= spatial_pos <= 400:
        return 'dashed'
    else:
        return 'solid'

# 在数据处理器中使用
processor.set_custom_interaction_rule(custom_interaction_rule)
```

### 2. 动态交互规则

```python
def dynamic_interaction_rule(lane_id, spatial_pos, time, traffic_density):
    """基于时间和交通密度的动态交互规则"""
    # 高峰时段限制交互
    if 7 <= time.hour <= 9 or 17 <= time.hour <= 19:
        return 'solid'
    
    # 高密度路段限制交互
    if traffic_density > 0.8:
        return 'solid'
    
    # 默认规则
    return 'dashed'
```

### 3. 多车道交互

```python
# 支持多车道同时交互
interaction_zones = [
    {
        'lanes': ['lane_0', 'lane_1', 'lane_2'],  # 三车道同时交互
        'start_pos': 0,
        'end_pos': 100,
        'type': 'dashed'
    }
]
```

## 可视化

### 1. 交互规则可视化

```python
import matplotlib.pyplot as plt

def visualize_interaction_rules(data):
    """可视化车道交互规则"""
    fig, axes = plt.subplots(len(lane_data), 1, figsize=(12, 8))
    
    for i, (lane_id, lane_df) in enumerate(lane_data.items()):
        ax = axes[i]
        
        # 绘制交互规则
        for _, row in lane_df.iterrows():
            pos = row['spatial_position']
            interaction = row['lane_interaction']
            color = 'orange' if interaction == 'dashed' else 'red'
            
            ax.barh(0, 10, left=pos, height=0.5, color=color, alpha=0.7)
        
        ax.set_title(f'{lane_id} 车道交互规则')
        ax.set_xlabel('位置 (米)')
    
    plt.tight_layout()
    plt.show()
```

### 2. 图连接可视化

```python
import networkx as nx

def visualize_graph_connections(adj_matrix, spatial_ids, lane_mapping):
    """可视化图连接"""
    G = nx.from_numpy_array(adj_matrix)
    
    # 按车道着色
    node_colors = []
    for node in G.nodes():
        spatial_id = spatial_ids[node]
        lane_id = lane_mapping[spatial_id]
        node_colors.append(lane_id)
    
    # 绘制图
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=node_colors, with_labels=True)
    plt.show()
```

## 性能优化

### 1. 连接矩阵优化

```python
# 使用稀疏矩阵存储连接
from scipy.sparse import csr_matrix

def build_sparse_adjacency(adj_matrix):
    """构建稀疏邻接矩阵"""
    return csr_matrix(adj_matrix)
```

### 2. 规则缓存

```python
# 缓存交互规则判断结果
interaction_cache = {}

def cached_can_interact(space1, space2):
    """缓存的交互判断"""
    key = (space1, space2)
    if key not in interaction_cache:
        interaction_cache[key] = can_lanes_interact(space1, space2)
    return interaction_cache[key]
```

## 故障排除

### 1. 常见问题

**问题：车道间没有连接**
- 检查交互规则列是否存在
- 验证交互规则值是否正确
- 确认两个车道在相同位置都允许交互

**问题：连接过多**
- 检查交互规则是否过于宽松
- 验证数据中是否有重复的交互规则

**问题：性能问题**
- 使用稀疏矩阵存储连接
- 启用规则缓存
- 减少不必要的连接检查

### 2. 调试工具

```python
def debug_interaction_rules(dataset):
    """调试交互规则"""
    print("车道交互规则调试信息:")
    
    # 统计交互规则分布
    rule_counts = dataset.df['lane_interaction'].value_counts()
    print(f"交互规则分布: {rule_counts.to_dict()}")
    
    # 检查图连接
    adj = dataset.get_connectivity()
    print(f"总连接数: {adj.sum()}")
    
    # 分析车道间连接
    lane_connections = analyze_lane_connections(adj, dataset)
    print(f"车道间连接: {lane_connections}")
```

## 最佳实践

### 1. 数据准备
- 确保交互规则列的数据完整性
- 使用标准化的交互规则值
- 验证交互规则的空间一致性

### 2. 规则设计
- 基于真实道路标线设计规则
- 考虑交通流量和安全性
- 支持动态规则更新

### 3. 性能考虑
- 合理设置交互区域大小
- 避免过于复杂的规则
- 使用缓存优化重复计算

## 示例代码

完整的使用示例请参考：
- `examples/lane_interaction_example.py` - 基本使用示例
- `examples/lane_traffic_example.py` - 完整流程示例
- `test_lane_dataset.py` - 测试脚本

