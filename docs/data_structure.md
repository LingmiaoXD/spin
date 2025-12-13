# 数据结构说明文档

## 概述

本系统采用**静态道路数据**和**动态交通数据**分离的设计，使得数据结构更加清晰，存储更加高效。

## 数据分类

### 1. 静态道路数据（Static Road Data）

静态道路数据包含不随时间变化的道路结构信息，主要用于定义空间拓扑关系。

`node_connections` 记录有向边和连接的节点，其中连接的节点用 `lane_id` 表示，有向边类型包含三种情况：
1. **前后直联（direct）**：当前节点与连接的节点属于同一车道。按车流方向，边权重为1
2. **相邻车道（near）**：当前节点与连接的节点属于相邻车道，边权重根据统计的变道概率计算
3. **交叉路口对应（crossing）**：当前节点需要通过一个红绿灯控制的路口才能到连接的节点。边权重可能为0/0.5/1，根据红绿灯情况实时调整

#### 数据字段

| 字段名 | 类型 | 描述 | 示例 |
|--------|------|------|------|
| `lane_id` | int | 车道标识符 | 0, 1, 2 |
| `node_connections` | dict | 节点连接规则 | {"direct": [2], "near": [3], "crossing": [1, 5]} |

#### 数据格式示例

**JSON格式（推荐）：**
```json
{
  "nodes": [
    {
      "lane_id": 0,
      "node_connections": {
        "direct": [1],
        "near": [3]
      }
    },
    {
      "lane_id": 1,
      "node_connections": {
        "direct": [0, 2],
        "near": [4]
      }
    },
    {
      "lane_id": 2,
      "node_connections": {
        "near": [1],
        "crossing": [4, 5]
      }
    }
  ]
}
```

### 2. 动态交通数据（Dynamic Traffic Data）

动态交通数据包含随时间变化的交通状况信息。

#### 数据字段

| 字段名 | 类型 | 描述 | 单位 | 示例 |
|--------|------|------|------|------|
| `lane_id` | int | 车道标识符 | - | 0 |
| `start_frame` | float | 起始帧/时间戳 | - | 0.0, 10.0 |
| `avg_speed` | float | 平均速度 | km/h | 0.03 |
| `avg_occupancy` | float | 平均占有率 | - | 0.4 |
| `total_vehicles` | float | 车辆总数（归一化） | - | 0.26 |
| `car_ratio` | float | 小汽车比例 | - | 1.0 |
| `medium_ratio` | float | 中型车比例 | - | 0.0 |
| `heavy_ratio` | float | 重型车比例 | - | 0.0 |
| `motorcycle_ratio` | float | 摩托车比例 | - | 0.0 |

> **注意**：`-1.0` 表示该字段不适用

#### 数据格式示例

**CSV格式（推荐）：**
```csv
lane_id,start_frame,avg_speed,avg_occupancy,total_vehicles,car_ratio,medium_ratio,heavy_ratio,motorcycle_ratio
0,0.0,0.0,0.4,0.26,1.0,0.0,0.0,0.0
0,10.0,0.0,0.4,0.26,1.0,0.0,0.0,0.0
0,20.0,0.03,0.4,0.26,1.0,0.0,0.0,0.0
1,0.0,0.05,0.35,0.30,0.8,0.1,0.1,0.0
1,10.0,0.08,0.38,0.32,0.75,0.15,0.1,0.0
```

## 数据关系

### 关联关系

- **静态道路数据**和**动态交通数据**通过 `lane_id` 字段关联
- 每个 `lane_id` 在静态数据中**有且仅有一条记录**
- 每个 `lane_id` 在动态数据中**每个时间步有一条记录**

### 数据关系图

```
静态道路数据 (static_road_data.json)
┌─────────────────────────────────────┐
│ lane_id | node_connections          │
├─────────────────────────────────────┤
│ 0       | {direct:[1], near:[3]}    │◄──┐
│ 1       | {direct:[0,2], near:[4]}  │   │
│ 2       | {near:[1], crossing:[4,5]}│   │ lane_id 关联
└─────────────────────────────────────┘   │
                                          │
动态交通数据 (dynamic_traffic_data.csv)   │
┌──────────────────────────────────────┐  │
│ lane_id | start_frame | avg_speed...│   │
├──────────────────────────────────────┤  │
│ 0       | 0.0         | 0.0    ...  │───┘
│ 0       | 10.0        | 0.0    ...  │
│ 1       | 0.0         | 0.05   ...  │
│ 1       | 10.0        | 0.08   ...  │
└──────────────────────────────────────┘
```

## 使用方法

### 1. 基本使用

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset
import json

# 加载静态道路数据（JSON格式）
with open("static_road_data.json", "r") as f:
    static_data = json.load(f)

# 加载动态交通数据（CSV格式）
dataset = LaneTrafficDataset(
    static_data_path="static_road_data.json",
    dynamic_data_path="dynamic_traffic_data.csv",
    window_size=12,
    stride=1
)

# 获取图连接矩阵（从静态数据构建）
adj = dataset.get_connectivity()

# 获取交通数据（从动态数据获取）
data = dataset.numpy()
```

### 2. 创建数据

```python
import json
import pandas as pd

# 创建静态道路数据（JSON格式）
static_data = {
    "nodes": [
        {"lane_id": 0, "node_connections": {"direct": [1], "near": [3]}},
        {"lane_id": 1, "node_connections": {"direct": [0, 2], "near": [4]}},
        {"lane_id": 2, "node_connections": {"near": [1], "crossing": [4, 5]}}
    ]
}

with open("static_road_data.json", "w") as f:
    json.dump(static_data, f, indent=2)

# 创建动态交通数据（CSV格式）
dynamic_data = pd.DataFrame([
    {"lane_id": 0, "start_frame": 0.0, "avg_speed": 0.0, "avg_occupancy": 0.4, 
     "total_vehicles": 0.26, "car_ratio": 1.0, "medium_ratio": 0.0, 
     "heavy_ratio": 0.0, "motorcycle_ratio": 0.0},
    {"lane_id": 0, "start_frame": 10.0, "avg_speed": 0.03, "avg_occupancy": 0.4, 
     "total_vehicles": 0.26, "car_ratio": 1.0, "medium_ratio": 0.0, 
     "heavy_ratio": 0.0, "motorcycle_ratio": 0.0}
])

dynamic_data.to_csv("dynamic_traffic_data.csv", index=False)
```

## 连接规则详解

### 连接类型

| 类型 | 描述 | 图连接权重 | 使用场景 |
|------|------|-----------|----------|
| `direct` | 直通连接 | 1.0 | 同一车道内相邻节点 |
| `near` | 相邻车道连接 | 根据变道概率 | 允许变道的跨车道连接 |
| `crossing` | 交叉路口连接 | 0/0.5/1（动态） | 红绿灯控制的路口 |

### 连接规则格式（JSON）

```json
{
  "direct": [1, 2],
  "near": [3, 4],
  "crossing": [5, 6]
}
```

## 配置文件示例

```yaml
# config/imputation/spin_lane.yaml
dataset:
  name: lane_traffic
  static_data_path: data/static_road_data.json
  dynamic_data_path: data/dynamic_traffic_data.csv
  
  # 动态数据列名配置
  dynamic_cols:
    lane_id: 'lane_id'
    start_frame: 'start_frame'
    avg_speed: 'avg_speed'
    avg_occupancy: 'avg_occupancy'
    total_vehicles: 'total_vehicles'
    car_ratio: 'car_ratio'
    medium_ratio: 'medium_ratio'
    heavy_ratio: 'heavy_ratio'
    motorcycle_ratio: 'motorcycle_ratio'
  
  # 数据处理参数
  window_size: 12
  stride: 1
  impute_nans: true
```

## 最佳实践

### 1. 数据格式选择

- **静态道路数据**：使用 JSON 格式，便于表示嵌套的连接关系
- **动态交通数据**：使用 CSV 格式，便于大规模时序数据存储和处理

### 2. 数据更新

- 静态数据应该保持稳定，仅在道路结构变化时更新
- 动态数据可以持续追加新的时间步数据
- 使用版本控制管理静态数据的变更

### 3. 数据验证

- 在加载数据前进行验证
- 确保静态数据和动态数据的 `lane_id` 一致
- 检查连接规则的完整性和正确性

## 示例数据集

项目提供了示例数据集供测试使用：

```bash
# 生成示例数据
python examples/create_sample_data.py

# 生成的文件
data/
  ├── static_road_data.json     # 静态道路数据（JSON格式）
  └── dynamic_traffic_data.csv  # 动态交通数据（CSV格式）
```

## 总结

采用静态和动态数据分离的设计具有以下优势：

✅ **清晰的数据结构** - 静态和动态信息明确分离  
✅ **高效的存储** - JSON存储拓扑结构，CSV存储时序数据  
✅ **便于维护** - 道路结构更新无需修改全部数据  
✅ **灵活性强** - 可以独立更新静态或动态数据  
✅ **性能优化** - 减少数据读取和内存占用
