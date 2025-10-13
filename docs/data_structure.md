# 数据结构说明文档

## 概述

本系统采用**静态道路数据**和**动态交通数据**分离的设计，使得数据结构更加清晰，存储更加高效。

## 数据分类

### 1. 静态道路数据（Static Road Data）

静态道路数据包含不随时间变化的道路结构信息，主要用于定义空间拓扑关系。

#### 数据字段

| 字段名 | 类型 | 描述 | 示例 |
|--------|------|------|------|
| `lane_id` | string | 车道标识符 | "lane_0", "lane_1" |
| `spatial_id` | string | 空间网格标识符 | "lane_0_0000", "lane_0_0001" |
| `node_connections` | string/dict | 节点连接规则 | "lane_0_0001,direct;lane_1_0000,dashed" |

#### 数据格式示例

**CSV格式：**
```csv
lane_id,spatial_id,node_connections
lane_0,lane_0_0000,"lane_0_0001,direct;lane_1_0000,dashed"
lane_0,lane_0_0001,"lane_0_0000,direct;lane_0_0002,direct;lane_1_0001,dashed"
lane_1,lane_1_0000,"lane_1_0001,direct;lane_0_0000,dashed"
lane_1,lane_1_0001,"lane_1_0000,direct;lane_1_0002,direct;lane_0_0001,dashed"
```

**JSON格式：**
```json
{
  "nodes": [
    {
      "lane_id": "lane_0",
      "spatial_id": "lane_0_0000",
      "node_connections": {
        "lane_0_0001": "direct",
        "lane_1_0000": "dashed"
      }
    },
    {
      "lane_id": "lane_0",
      "spatial_id": "lane_0_0001",
      "node_connections": {
        "lane_0_0000": "direct",
        "lane_0_0002": "direct",
        "lane_1_0001": "dashed"
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
| `timestamp` | datetime | 时间戳 | - | "2024-01-01 00:00:00" |
| `spatial_id` | string | 空间网格标识符 | - | "lane_0_0000" |
| `speed` | float | 平均速度 | km/h | 30.5 |
| `spacing` | float | 平均间距 | m | 25.2 |

#### 数据格式示例

**CSV格式：**
```csv
timestamp,spatial_id,speed,spacing
2024-01-01 00:00:00,lane_0_0000,30.5,25.2
2024-01-01 00:00:00,lane_0_0001,32.1,23.8
2024-01-01 00:00:00,lane_1_0000,28.9,26.5
2024-01-01 00:00:10,lane_0_0000,31.2,24.8
2024-01-01 00:00:10,lane_0_0001,33.0,23.5
```

**NPZ格式：**
```python
{
    'timestamps': np.array([...]),    # 时间戳数组
    'spatial_ids': np.array([...]),   # 空间ID数组
    'speeds': np.array([...]),        # 速度数组
    'spacings': np.array([...])       # 间距数组
}
```

## 数据关系

### 关联关系

- **静态道路数据**和**动态交通数据**通过 `spatial_id` 字段关联
- 每个 `spatial_id` 在静态数据中**有且仅有一条记录**
- 每个 `spatial_id` 在动态数据中**每个时间步有一条记录**

### 数据关系图

```
静态道路数据 (static_road_data.csv)
┌─────────────────────────────────────────┐
│ lane_id | spatial_id | node_connections │
├─────────────────────────────────────────┤
│ lane_0  | lane_0_0000| ...              │◄──┐
│ lane_0  | lane_0_0001| ...              │   │
│ lane_1  | lane_1_0000| ...              │   │ spatial_id 关联
└─────────────────────────────────────────┘   │
                                              │
动态交通数据 (dynamic_traffic_data.csv)       │
┌──────────────────────────────────────┐      │
│ timestamp | spatial_id | speed | ... │      │
├──────────────────────────────────────┤      │
│ 00:00:00  | lane_0_0000| 30.5  | ... │──────┘
│ 00:00:00  | lane_0_0001| 32.1  | ... │
│ 00:00:10  | lane_0_0000| 31.2  | ... │
└──────────────────────────────────────┘
```

## 使用方法

### 1. 基本使用

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset

# 加载分离的数据
dataset = LaneTrafficDataset(
    static_data_path="static_road_data.csv",      # 静态道路数据
    dynamic_data_path="dynamic_traffic_data.csv",  # 动态交通数据
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
from spin.datasets.lane_data_utils import LaneDataProcessor

processor = LaneDataProcessor()

# 生成静态道路数据
static_data = processor.create_static_road_data(
    n_lanes=3,
    lane_length=1000.0
)
processor.save_data(static_data, "static_road_data.csv", format='csv')

# 生成动态交通数据
dynamic_data = processor.create_dynamic_traffic_data(
    static_data=static_data,
    time_hours=24.0
)
processor.save_data(dynamic_data, "dynamic_traffic_data.csv", format='csv')
```

### 3. 处理原始数据

```python
# 从原始数据生成分离的数据文件
raw_data = pd.read_csv("raw_traffic_data.csv")

# 提取静态道路数据
static_data = processor.extract_static_data(raw_data)
processor.save_data(static_data, "static_road_data.csv")

# 提取动态交通数据
dynamic_data = processor.extract_dynamic_data(raw_data)
processor.save_data(dynamic_data, "dynamic_traffic_data.csv")
```

## 数据验证

### 静态数据验证

```python
from spin.datasets.lane_data_utils import validate_static_data

# 验证静态道路数据
is_valid, errors = validate_static_data(static_data)
if is_valid:
    print("✅ 静态数据验证通过")
else:
    print(f"❌ 静态数据验证失败: {errors}")
```

验证项目：
- 必需字段存在性检查
- `spatial_id` 唯一性检查
- `node_connections` 格式验证
- 连接目标节点存在性验证

### 动态数据验证

```python
from spin.datasets.lane_data_utils import validate_dynamic_data

# 验证动态交通数据
is_valid, errors = validate_dynamic_data(dynamic_data, static_data)
if is_valid:
    print("✅ 动态数据验证通过")
else:
    print(f"❌ 动态数据验证失败: {errors}")
```

验证项目：
- 必需字段存在性检查
- 时间戳格式验证
- `spatial_id` 与静态数据一致性检查
- 数值范围合理性检查
- 缺失值统计

## 连接规则详解

### 连接类型

| 类型 | 描述 | 图连接权重 | 使用场景 |
|------|------|-----------|----------|
| `direct` | 直通连接 | 1.0 | 同一车道内相邻节点 |
| `dashed` | 虚线连接 | 1.0 | 允许变道的跨车道连接 |
| `solid` | 实线连接 | 0.0 | 禁止变道的跨车道分隔 |

### 连接规则格式

**字符串格式（推荐）：**
```
"target1,type1;target2,type2;target3,type3"
```

**示例：**
```
"lane_0_0001,direct;lane_1_0000,dashed;lane_2_0032,dashed"
```

**字典格式：**
```python
{
    "lane_0_0001": "direct",
    "lane_1_0000": "dashed",
    "lane_2_0032": "dashed"
}
```

**JSON格式：**
```json
{"lane_0_0001": "direct", "lane_1_0000": "dashed", "lane_2_0032": "dashed"}
```

## 性能优化

### 存储优化

分离数据设计的优势：

1. **减少冗余**：静态数据只存储一次，不在每个时间步重复
   - 传统方式：`n_timestamps × n_nodes` 条静态数据记录
   - 分离方式：`n_nodes` 条静态数据记录
   - **节省比例**：约 `(n_timestamps - 1) / n_timestamps` （例如1000个时间步节省99.9%）

2. **提高读取效率**：只需读取一次静态数据
3. **便于更新**：修改道路结构无需更新所有时间步的数据

### 内存优化

```python
# 使用预处理缓存
dataset = LaneTrafficDataset(
    static_data_path="static_road_data.csv",
    dynamic_data_path="dynamic_traffic_data.csv",
    use_cache=True,                    # 启用缓存
    cache_dir="./cache"                # 缓存目录
)
```

## 数据迁移指南

### 从旧格式迁移到新格式

如果您有旧格式的数据（静态和动态信息混合），可以使用以下方法迁移：

```python
from spin.datasets.lane_data_utils import migrate_to_separated_format

# 读取旧格式数据
old_data = pd.read_csv("old_lane_data.csv")

# 迁移到新格式
static_data, dynamic_data = migrate_to_separated_format(
    old_data,
    static_cols=['lane_id', 'spatial_id', 'node_connections'],
    dynamic_cols=['timestamp', 'spatial_id', 'speed', 'spacing']
)

# 保存新格式数据
static_data.to_csv("static_road_data.csv", index=False)
dynamic_data.to_csv("dynamic_traffic_data.csv", index=False)
```

## 配置文件示例

```yaml
# config/imputation/spin_lane.yaml
dataset:
  name: lane_traffic
  static_data_path: data/static_road_data.csv
  dynamic_data_path: data/dynamic_traffic_data.csv
  
  # 数据列名配置
  static_cols:
    lane_id: 'lane_id'
    spatial_id: 'spatial_id'
    node_connections: 'node_connections'
  
  dynamic_cols:
    timestamp: 'timestamp'
    spatial_id: 'spatial_id'
    speed: 'speed'
    spacing: 'spacing'
  
  # 数据处理参数
  window_size: 12
  stride: 1
  impute_nans: true
```

## 最佳实践

### 1. 数据组织

- 将静态数据和动态数据放在不同的文件中
- 使用清晰的文件命名：`static_road_data.csv`, `dynamic_traffic_data.csv`
- 为不同数据集创建独立的目录

### 2. 数据更新

- 静态数据应该保持稳定，仅在道路结构变化时更新
- 动态数据可以持续追加新的时间步数据
- 使用版本控制管理静态数据的变更

### 3. 数据验证

- 在加载数据前进行验证
- 确保静态数据和动态数据的 `spatial_id` 一致
- 检查连接规则的完整性和正确性

### 4. 性能考虑

- 对于大规模数据，考虑使用 HDF5 或 Parquet 格式
- 启用数据缓存减少重复读取
- 使用数据分块处理超大数据集

## 示例数据集

项目提供了示例数据集供测试使用：

```bash
# 生成示例数据
python examples/create_sample_data.py

# 生成的文件
data/
  ├── static_road_data.csv      # 静态道路数据
  └── dynamic_traffic_data.csv  # 动态交通数据
```

## 技术支持

如有问题，请参考：
1. `examples/` 目录中的示例代码
2. `docs/` 目录中的详细文档
3. 项目 README 文件

## 总结

采用静态和动态数据分离的设计具有以下优势：

✅ **清晰的数据结构** - 静态和动态信息明确分离  
✅ **高效的存储** - 避免静态信息的重复存储  
✅ **便于维护** - 道路结构更新无需修改全部数据  
✅ **灵活性强** - 可以独立更新静态或动态数据  
✅ **性能优化** - 减少数据读取和内存占用

