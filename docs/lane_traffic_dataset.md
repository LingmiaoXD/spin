# 车道级交通数据集使用指南

本指南介绍如何使用车道级交通状况数据集进行时空数据插补。

## 数据集概述

车道级交通数据集专门设计用于处理10m×10s时空网格内的交通状况数据：

- **空间分辨率**: 10米
- **时间分辨率**: 10秒  
- **节点属性**: 平均速度、平均间距
- **图连接**: 基于节点连接规则的空间关系
- **数据格式**: 静态道路数据 + 动态交通数据分离

## 数据格式

### 静态 + 动态分离格式

数据集采用**静态道路数据**和**动态交通数据**分离的设计：

#### 1. 静态道路数据 (Static Road Data)

包含不随时间变化的道路结构信息。

**CSV格式示例：**
```csv
lane_id,spatial_id,node_connections
lane_0,lane_0_0000,"lane_0_0001,direct;lane_1_0000,dashed"
lane_0,lane_0_0001,"lane_0_0000,direct;lane_0_0002,direct"
lane_1,lane_1_0000,"lane_1_0001,direct;lane_0_0000,dashed"
```

**字段说明：**

| 字段名 | 类型 | 描述 |
|--------|------|------|
| lane_id | string | 车道标识符 |
| spatial_id | string | 空间网格标识符（唯一） |
| node_connections | string | 节点连接规则 |

#### 2. 动态交通数据 (Dynamic Traffic Data)

包含随时间变化的交通状况信息。

**CSV格式示例：**
```csv
timestamp,spatial_id,speed,spacing
2024-01-01 00:00:00,lane_0_0000,30.5,25.2
2024-01-01 00:00:00,lane_0_0001,32.1,23.8
2024-01-01 00:00:10,lane_0_0000,31.2,24.8
```

**NPZ格式：**
```python
{
    'timestamps': np.array([...]),      # 时间戳数组
    'spatial_ids': np.array([...]),     # 空间ID数组
    'speeds': np.array([...]),          # 速度数组
    'spacings': np.array([...])         # 间距数组
}
```

**字段说明：**

| 字段名 | 类型 | 描述 |
|--------|------|------|
| timestamp | datetime | 时间戳 |
| spatial_id | string | 空间网格标识符 |
| speed | float | 平均速度 (km/h) |
| spacing | float | 平均间距 (m) |

## 快速开始

### 1. 创建示例数据

```python
from spin.datasets.lane_data_utils import create_separated_sample_dataset

# 创建分离格式的示例数据集
static_data, dynamic_data = create_separated_sample_dataset(
    static_output_path="static_road_data.csv",
    dynamic_output_path="dynamic_traffic_data.csv"
)
```

### 2. 加载数据集

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset

# 加载车道数据集（使用分离格式）
dataset = LaneTrafficDataset(
    static_data_path="static_road_data.csv",
    dynamic_data_path="dynamic_traffic_data.csv",
    impute_nans=True,
    window_size=12,
    stride=1
)
```

### 3. 运行训练

```bash
python experiments/run_imputation.py \
    --model-name spin \
    --dataset-name lane_traffic \
    --static-data-path static_road_data.csv \
    --dynamic-data-path dynamic_traffic_data.csv \
    --config config/imputation/spin_lane.yaml \
    --epochs 100 \
    --batch-size 8
```

## 详细使用

### 数据预处理

使用 `LaneDataProcessor` 类生成分离格式的数据：

```python
from spin.datasets.lane_data_utils import LaneDataProcessor

# 创建数据处理器
processor = LaneDataProcessor(
    spatial_resolution=10.0,  # 10米空间分辨率
    temporal_resolution=10    # 10秒时间分辨率
)

# 创建静态道路数据
static_data = processor.create_static_road_data(
    n_lanes=3,
    lane_length=1000.0,
    seed=42
)

# 创建动态交通数据
dynamic_data = processor.create_dynamic_traffic_data(
    static_data=static_data,
    time_hours=24.0,
    seed=42
)

# 保存数据
processor.save_data(static_data, "static_road_data.csv", format='csv')
processor.save_data(dynamic_data, "dynamic_traffic_data.csv", format='csv')
```

### 自定义车道信息

```python
# 创建车道信息字典
lane_info = processor.create_lane_info(
    lane_ids=['lane_0', 'lane_1', 'lane_2'],
    lane_lengths=[1000.0, 1200.0, 800.0],
    lane_positions=[(0, 0), (0, 3.5), (0, 7.0)]
)
```

### 数据集配置

车道数据集支持以下配置参数：

```yaml
# config/imputation/spin_lane.yaml
dataset_name: lane_traffic
static_data_path: data/static_road_data.csv
dynamic_data_path: data/dynamic_traffic_data.csv

# 数据列名配置
speed_col: 'speed'              # 速度列名
spacing_col: 'spacing'          # 间距列名
time_col: 'timestamp'           # 时间列名
lane_id_col: 'lane_id'          # 车道ID列名
spatial_id_col: 'spatial_id'    # 空间ID列名
node_connections_col: 'node_connections'  # 节点连接列名

# 数据处理参数
window_size: 12
stride: 1
impute_nans: true
```

## 图连接策略

### 1. 车道内连接
同一车道内的相邻空间网格节点自动连接，形成链式结构（direct连接）。

### 2. 跨车道连接
基于节点连接规则 (`node_connections`) 的跨车道连接，支持：
- `direct`: 直通连接（同车道相邻节点）
- `dashed`: 虚线连接（允许变道的跨车道连接）
- `solid`: 实线连接（禁止连接）

### 3. 连接权重
- direct连接权重: 1.0
- dashed连接权重: 1.0
- solid连接权重: 0.0（不建立连接）

## 数据验证

使用内置的数据验证功能检查数据质量：

```python
# 验证数据格式和完整性
if processor.validate_data(data):
    print("✅ 数据验证通过")
else:
    print("❌ 数据验证失败")
```
