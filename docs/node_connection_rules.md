# 节点级连接规则实现指南

## 概述

节点级连接规则允许为每个具体的节点指定与其他节点的连接关系，实现精确的图连接控制。这比车道级交互规则更加灵活和精确。

## 数据格式

### 输入数据格式

在原始数据中添加 `node_connections` 列来指定每个节点的连接规则：

```csv
timestamp,lane_id,spatial_id,speed,spacing,node_connections
2024-01-01 00:00:00,lane_0,lane_0_0000,30.5,25.2,"lane_0_0001,direct;lane_1_0000,dashed;lane_2_0032,dashed"
2024-01-01 00:00:00,lane_0,lane_0_0001,32.1,23.8,"lane_0_0000,direct;lane_0_0002,direct;lane_1_0001,dashed"
2024-01-01 00:00:00,lane_1,lane_1_0000,28.9,26.5,"lane_1_0001,direct;lane_0_0000,dashed;lane_2_0000,dashed"
```

### 连接规则格式

**字符串格式：**
```
"target1,type1;target2,type2;target3,type3"
```

**示例：**
```
"lane_0_0001,direct;lane_1_0000,dashed;lane_2_0032,dashed"
```

### 连接类型

| 类型 | 描述 | 图连接 | 说明 |
|------|------|--------|------|
| `direct` | 直通连接 | ✅ 建立连接 | 同一车道内相邻节点 |
| `dashed` | 虚线连接 | ✅ 建立连接 | 跨车道连接 |
| `solid` | 实线连接 | ❌ 不建立连接 | 禁止连接 |

## 实现原理

### 1. 图连接构建流程

```python
def _build_graph_connectivity(self):
    # 1. 车道内连接（纵向）
    for lane_id in self.lane_ids:
        connect_lane_internal_nodes(lane_id)
    
    # 2. 节点级连接（基于连接规则）
    for spatial_id, connections in spatial_to_connections.items():
        parse_and_apply_connections(spatial_id, connections)
```

### 2. 连接规则解析

```python
def _parse_node_connections(self, connections):
    # 支持多种格式：
    # 1. 字符串: "lane_0_0001,direct;lane_1_0000,dashed"
    # 2. 字典: {"lane_0_0001": "direct", "lane_1_0000": "dashed"}
    # 3. JSON: '{"lane_0_0001": "direct", "lane_1_0000": "dashed"}'
```

### 3. 连接权重设置

```python
def _add_node_connections(self, adj_matrix, spatial_to_connections):
    for target_spatial_id, connection_type in connections.items():
        if connection_type in ['direct', 'dashed']:
            weight = 1.0  # 建立连接
        elif connection_type in ['solid']:
            weight = 0.0  # 不连接
```

## 使用方法

### 1. 基本使用

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset

# 加载包含节点连接规则的数据
dataset = LaneTrafficDataset(
    data_path="lane_data_with_node_connections.csv",
    node_connections_col='node_connections'
)

# 获取图连接矩阵
adj = dataset.get_connectivity()
```

### 2. 创建节点连接规则

```python
from spin.datasets.lane_data_utils import LaneDataProcessor

processor = LaneDataProcessor()

# 定义节点连接规则
node_connection_rules = [
    {
        'spatial_id': 'lane_0_0000',
        'connections': {
            'lane_0_0001': 'direct',    # 与 lane_0_0001 直通连接
            'lane_1_0000': 'dashed',    # 与 lane_1_0000 虚线连接
            'lane_2_0032': 'dashed'     # 与 lane_2_0032 虚线连接
        }
    }
]

# 创建节点连接规则
node_rules = processor.create_node_connection_rules(
    spatial_ids=['lane_0_0000', 'lane_0_0001'],
    connection_rules=node_connection_rules
)
```

### 3. 处理原始数据

```python
# 处理包含节点连接规则的原始数据
processed_data = processor.process_raw_data(
    raw_data=your_raw_data,
    lane_info=lane_rules
)
```

## 高级配置

### 1. 自定义连接规则

```python
def custom_connection_rule(spatial_id, lane_id, spatial_pos, context):
    """自定义连接规则函数"""
    connections = {}
    
    # 基于位置的规则
    if spatial_pos < 100:
        connections[f"{lane_id}_0001"] = 'direct'
        connections[f"lane_1_{spatial_pos:04d}"] = 'dashed'
    
    return connections

# 在数据处理器中使用
processor.set_custom_connection_rule(custom_connection_rule)
```

### 2. 动态连接规则

```python
def dynamic_connection_rule(spatial_id, lane_id, spatial_pos, time, traffic_density):
    """基于时间和交通密度的动态连接规则"""
    connections = {}
    
    # 高峰时段限制跨车道连接
    if 7 <= time.hour <= 9 or 17 <= time.hour <= 19:
        # 只允许车道内连接
        connections[f"{lane_id}_0001"] = 'direct'
    else:
        # 允许跨车道连接
        connections[f"{lane_id}_0001"] = 'direct'
        connections[f"lane_1_{spatial_pos:04d}"] = 'dashed'
    
    return connections
```

### 3. 复杂连接模式

```python
# 支持复杂的连接模式
complex_connections = [
    {
        'spatial_id': 'lane_0_0000',
        'connections': {
            'lane_0_0001': 'direct',      # 前一个节点
            'lane_0_0002': 'direct',      # 后一个节点
            'lane_1_0000': 'dashed',      # 左侧车道
            'lane_1_0001': 'dashed',      # 左侧车道下一个
            'lane_2_0000': 'dashed',      # 右侧车道
            'lane_2_0001': 'dashed'       # 右侧车道下一个
        }
    }
]
```

## 数据格式示例

### 1. CSV格式

```csv
timestamp,lane_id,spatial_id,speed,spacing,node_connections
2024-01-01 00:00:00,lane_0,lane_0_0000,30.5,25.2,"lane_0_0001,direct;lane_1_0000,dashed"
2024-01-01 00:00:00,lane_0,lane_0_0001,32.1,23.8,"lane_0_0000,direct;lane_0_0002,direct;lane_1_0001,dashed"
2024-01-01 00:00:00,lane_1,lane_1_0000,28.9,26.5,"lane_1_0001,direct;lane_0_0000,dashed"
```

### 2. JSON格式

```json
{
  "timestamp": "2024-01-01 00:00:00",
  "lane_id": "lane_0",
  "spatial_id": "lane_0_0000",
  "speed": 30.5,
  "spacing": 25.2,
  "node_connections": "{\"lane_0_0001\": \"direct\", \"lane_1_0000\": \"dashed\"}"
}
```

### 3. 字典格式

```python
data = {
    'timestamp': '2024-01-01 00:00:00',
    'lane_id': 'lane_0',
    'spatial_id': 'lane_0_0000',
    'speed': 30.5,
    'spacing': 25.2,
    'node_connections': {
        'lane_0_0001': 'direct',
        'lane_1_0000': 'dashed'
    }
}
```

## 可视化

### 1. 连接规则可视化

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_node_connections(data):
    """可视化节点连接规则"""
    G = nx.Graph()
    
    # 添加节点
    for _, row in data.iterrows():
        G.add_node(row['spatial_id'], lane=row['lane_id'])
    
    # 添加边
    for _, row in data.iterrows():
        connections = row['node_connections']
        if pd.notna(connections):
            for connection in connections.split(';'):
                if ',' in connection:
                    target, conn_type = connection.strip().split(',', 1)
                    G.add_edge(row['spatial_id'], target, type=conn_type)
    
    # 绘制图
    pos = nx.spring_layout(G)
    
    # 按车道着色
    lane_colors = {'lane_0': 'red', 'lane_1': 'blue', 'lane_2': 'green'}
    node_colors = [lane_colors.get(G.nodes[node]['lane'], 'gray') for node in G.nodes()]
    
    # 按连接类型设置边样式
    direct_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'direct']
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'dashed']
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=direct_edges, edge_color='black', width=2, style='-')
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color='orange', width=2, style='--')
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('节点连接规则可视化')
    plt.show()
```

### 2. 连接矩阵可视化

```python
import seaborn as sns

def visualize_connection_matrix(adj_matrix, spatial_ids):
    """可视化连接矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(adj_matrix, 
                xticklabels=spatial_ids, 
                yticklabels=spatial_ids,
                cmap='Blues',
                cbar=True)
    plt.title('节点连接矩阵')
    plt.xlabel('目标节点')
    plt.ylabel('源节点')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
```

## 性能优化

### 1. 连接规则缓存

```python
# 缓存连接规则解析结果
connection_cache = {}

def cached_parse_connections(connections):
    """缓存的连接规则解析"""
    if connections not in connection_cache:
        connection_cache[connections] = parse_node_connections(connections)
    return connection_cache[connections]
```

### 2. 稀疏矩阵存储

```python
from scipy.sparse import csr_matrix

def build_sparse_adjacency(adj_matrix):
    """构建稀疏邻接矩阵"""
    return csr_matrix(adj_matrix)
```

### 3. 批量连接处理

```python
def batch_process_connections(connections_list):
    """批量处理连接规则"""
    results = []
    for connections in connections_list:
        results.append(parse_node_connections(connections))
    return results
```

## 故障排除

### 1. 常见问题

**问题：节点连接不生效**
- 检查 `node_connections` 列是否存在
- 验证连接规则格式是否正确
- 确认目标节点是否存在

**问题：连接过多或过少**
- 检查连接规则是否过于宽松或严格
- 验证连接类型设置是否正确
- 确认连接规则没有重复

**问题：性能问题**
- 使用稀疏矩阵存储连接
- 启用连接规则缓存
- 减少不必要的连接检查

### 2. 调试工具

```python
def debug_node_connections(dataset):
    """调试节点连接规则"""
    print("节点连接规则调试信息:")
    
    # 统计连接规则分布
    connection_counts = dataset.df['node_connections'].str.count(';') + 1
    print(f"平均连接数: {connection_counts.mean():.2f}")
    print(f"最大连接数: {connection_counts.max()}")
    
    # 检查图连接
    adj = dataset.get_connectivity()
    print(f"总连接数: {adj.sum()}")
    
    # 分析连接模式
    connection_patterns = analyze_connection_patterns(adj, dataset)
    print(f"连接模式: {connection_patterns}")
```

## 最佳实践

### 1. 数据准备
- 确保连接规则列的数据完整性
- 使用标准化的连接规则格式
- 验证目标节点的存在性

### 2. 规则设计
- 基于实际交通规则设计连接
- 考虑空间和时间的约束
- 支持动态规则更新

### 3. 性能考虑
- 合理设置连接数量
- 避免过于复杂的规则
- 使用缓存优化重复计算

## 示例代码

完整的使用示例请参考：
- `examples/node_connection_example.py` - 基本使用示例
- `test_node_connections.py` - 测试脚本
- `config/imputation/spin_lane.yaml` - 配置文件

## 总结

节点级连接规则提供了最灵活和精确的图连接控制方式，支持：

✅ **精确控制** - 为每个节点指定具体的连接关系
✅ **多种格式** - 支持字符串、字典、JSON等多种格式
✅ **灵活配置** - 支持自定义连接规则和动态规则
✅ **高性能** - 优化的解析和缓存机制
✅ **易于使用** - 简单的API和详细的文档

这种实现方式完全满足您提到的需求：lane_0_0000 可以与 lane_0_0001 直通连接，与 lane_1_0000 和 lane_2_0032 虚线连接。

