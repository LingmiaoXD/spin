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

验证项目包括：
- 必需字段检查
- 数据类型验证
- 数值范围检查
- 缺失值统计

## 性能优化

### 内存优化
- 使用预处理数据缓存
- 支持数据分块加载
- 可配置的批处理大小

### 计算优化
- 向量化数据处理
- 并行图连接计算
- 高效的时间编码

## 故障排除

### 常见问题

1. **数据格式错误**
   - 检查CSV列名是否匹配配置
   - 确保时间格式正确
   - 验证数值范围合理

2. **内存不足**
   - 减少批处理大小
   - 使用数据分块
   - 启用梯度检查点

3. **图连接问题**
   - 检查车道ID是否连续
   - 验证空间ID格式
   - 调整连接阈值

### 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行数据集加载
dataset = LaneTrafficDataset(data_path="your_data.csv", debug=True)
```

## 数据迁移

如果您有旧格式的混合数据，可以轻松迁移到新格式：

```python
from spin.datasets.lane_data_utils import migrate_to_separated_format
import pandas as pd

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

## 示例代码

完整的使用示例请参考：
- `examples/separated_data_example.py` - 分离格式数据使用示例
- `examples/lane_traffic_example.py` - 基本使用示例
- `experiments/run_imputation.py` - 训练脚本
- `config/imputation/spin_lane.yaml` - 配置文件
- `docs/data_structure.md` - 数据结构详细说明

## 技术支持

如有问题，请检查：
1. 数据格式是否符合要求
2. 配置文件参数是否正确
3. 依赖包版本是否兼容
4. 系统内存是否充足

