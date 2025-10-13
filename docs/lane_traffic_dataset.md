# 车道级交通数据集使用指南

本指南介绍如何使用车道级交通状况数据集进行时空数据插补。

## 数据集概述

车道级交通数据集专门设计用于处理10m×10s时空网格内的交通状况数据：

- **空间分辨率**: 10米
- **时间分辨率**: 10秒  
- **节点属性**: 平均速度、平均间距
- **图连接**: 基于车道关联的空间关系

## 数据格式

### 输入数据格式

数据集支持以下格式的输入文件：

#### CSV格式
```csv
timestamp,lane_id,spatial_id,speed,spacing
2024-01-01 00:00:00,lane_0,lane_0_0000,30.5,25.2
2024-01-01 00:00:10,lane_0,lane_0_0001,32.1,23.8
...
```

#### NPZ格式
```python
{
    'timestamps': np.array([...]),      # 时间戳数组
    'lane_ids': np.array([...]),        # 车道ID数组
    'spatial_ids': np.array([...]),     # 空间ID数组
    'speeds': np.array([...]),          # 速度数组
    'spacings': np.array([...])         # 间距数组
}
```

### 数据字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| timestamp | datetime | 时间戳 |
| lane_id | string | 车道标识符 |
| spatial_id | string | 空间网格标识符 |
| speed | float | 平均速度 (km/h) |
| spacing | float | 平均间距 (m) |

## 快速开始

### 1. 创建示例数据

```python
from spin.datasets.lane_data_utils import create_sample_dataset

# 创建示例数据集
sample_data = create_sample_dataset("sample_lane_data.csv")
```

### 2. 加载数据集

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset

# 加载车道数据集
dataset = LaneTrafficDataset(
    data_path="sample_lane_data.csv",
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
    --data-path sample_lane_data.csv \
    --config config/imputation/spin_lane.yaml \
    --epochs 100 \
    --batch-size 8
```

## 详细使用

### 数据预处理

使用 `LaneDataProcessor` 类处理原始车道数据：

```python
from spin.datasets.lane_data_utils import LaneDataProcessor

# 创建数据处理器
processor = LaneDataProcessor(
    spatial_resolution=10.0,  # 10米空间分辨率
    temporal_resolution=10,   # 10秒时间分辨率
    speed_col='speed',
    spacing_col='spacing',
    time_col='timestamp',
    lane_id_col='lane_id',
    spatial_id_col='spatial_id'
)

# 处理原始数据
processed_data = processor.process_raw_data(raw_data, lane_info)

# 保存处理后的数据
processor.save_data(processed_data, "processed_data.csv", format='csv')
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
data_path: sample_lane_data.csv

# 车道级数据特有参数
spatial_resolution: 10.0    # 空间分辨率（米）
temporal_resolution: 10     # 时间分辨率（秒）
speed_col: 'speed'          # 速度列名
spacing_col: 'spacing'      # 间距列名
time_col: 'timestamp'       # 时间列名
lane_id_col: 'lane_id'      # 车道ID列名
spatial_id_col: 'spatial_id' # 空间ID列名

# 图连接策略
connectivity_strategy: 'lane_based'  # 基于车道的连接
cross_lane_connection: True          # 启用跨车道连接
max_cross_lane_distance: 50.0       # 跨车道连接最大距离
```

## 图连接策略

### 1. 车道内连接
同一车道内的相邻空间网格节点相互连接，形成链式结构。

### 2. 跨车道连接
基于空间距离的跨车道连接，用于捕获车道间的相互影响。

### 3. 连接权重
- 车道内连接权重: 1.0
- 跨车道连接权重: 基于距离的衰减函数

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

## 示例代码

完整的使用示例请参考：
- `examples/lane_traffic_example.py` - 基本使用示例
- `experiments/run_imputation.py` - 训练脚本
- `config/imputation/spin_lane.yaml` - 配置文件

## 技术支持

如有问题，请检查：
1. 数据格式是否符合要求
2. 配置文件参数是否正确
3. 依赖包版本是否兼容
4. 系统内存是否充足

