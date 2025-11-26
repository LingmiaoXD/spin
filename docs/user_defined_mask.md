# 用户自定义掩码功能说明

## 概述

数据集现在支持用户自定义掩码，允许用户精确控制哪些时间窗口下哪些lane_id的值是已知的（已观测），哪些是未知的（需要预测）。

## 核心概念

- **已知数据（已观测）**：在掩码中标记为 `True` 的数据点，表示这些值是已知的，可以用于训练
- **未知数据（未观测）**：在掩码中标记为 `False` 的数据点，表示这些值是未知的，需要模型进行预测

## 使用方法

### 1. 基本用法

在创建 `LaneTrafficDataset` 时，通过 `mask_data_path` 参数指定掩码文件路径：

```python
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset

dataset = LaneTrafficDataset(
    static_data_path='data/static_road_data.csv',
    dynamic_data_path='data/dynamic_traffic_data.csv',
    mask_data_path='data/mask.csv',  # 指定掩码文件
    time_col='start_frame',
    spatial_id_col='lane_id',
    speed_col='speed',
    spacing_col='spacing'
)
```

如果不提供 `mask_data_path` 参数，数据集将使用默认的随机掩码生成方式。

### 2. 支持的掩码文件格式

#### 2.1 CSV格式

CSV格式是最灵活和直观的方式，适合精确控制每个时间窗口、每个lane_id的观测状态。

**文件结构：**

| start_frame | lane_id | is_observed |
|-----------|------------|-------------|
| 0 | 1 | 1 |
| 10 | 2 | 0 |
| 20 | 1 | 1 |
| ... | ... | ... |

**必需列：**
- `start_frame`: 各个时间窗口的起始时间（与动态数据中的时间列对应）
- `lane_id`: 路段ID（与动态数据中的路段ID列对应）
- `is_observed`: 是否已观测（1=已知，0=未知）

**示例代码：**

```python
import pandas as pd

# 创建掩码数据
mask_data = []
start_frames = range(0, 300, 10)  # 时间窗口起始帧：0, 10, 20, ...
lane_ids = [1, 2, 3, 4]

for start_frame in start_frames:
    for lane_id in lane_ids:
        # 只有lane_id为1和3的是已知的
        is_observed = lane_id in [1, 3]
        mask_data.append({
            'start_frame': start_frame,
            'lane_id': lane_id,
            'is_observed': int(is_observed)
        })

# 保存为CSV
mask_df = pd.DataFrame(mask_data)
mask_df.to_csv('mask.csv', index=False)
```

#### 2.2 NPZ格式

NPZ格式适合直接提供掩码矩阵，更加简洁高效。

**文件结构：**
- 必须包含名为 `'mask'` 的数组
- 形状：`[n_times, n_spaces]` 或 `[n_times, n_spaces, n_features]`
- 值：`True`=已知，`False`=未知

**示例代码：**

```python
import numpy as np

# 创建掩码矩阵
n_times = 30  # 时间步数
n_spaces = 4  # 空间节点数

# 在每个时间步，只有前2个空间节点是已知的
mask = np.zeros((n_times, n_spaces), dtype=bool)
mask[:, :2] = True

# 保存为NPZ
np.savez('mask.npz', mask=mask)
```

**注意：** 
- 如果提供二维掩码 `[n_times, n_spaces]`，该掩码将应用于所有特征（speed, spacing等）
- 如果提供三维掩码 `[n_times, n_spaces, n_features]`，可以为不同特征设置不同的掩码

#### 2.3 PKL格式

PKL格式支持Python对象序列化，可以直接保存numpy数组或字典。

**方式1：直接保存掩码矩阵**

```python
import pickle
import numpy as np

# 创建掩码矩阵
mask = np.zeros((30, 4), dtype=bool)
mask[:, :2] = True

# 保存为PKL
with open('mask.pkl', 'wb') as f:
    pickle.dump(mask, f)
```

**方式2：保存字典**

```python
import pickle
import numpy as np

# 创建掩码数据
mask_data = {
    'training_mask': np.zeros((30, 4, 2), dtype=bool)
}
mask_data['training_mask'][:, :2, :] = True

# 保存为PKL
with open('mask.pkl', 'wb') as f:
    pickle.dump(mask_data, f)
```

**注意：** 如果使用字典格式，必须包含 `'training_mask'` 键。

## 应用场景

### 场景2：传感器故障

某些时间段某些传感器发生故障，导致数据缺失。

```python
# 在特定时间段，某些传感器故障
faulty_lanes = [2, 4]
fault_start_frame = 100
fault_end_frame = 200

mask_data = []
for start_frame in all_start_frames:
    for lane_id in all_lane_ids:
        # 在故障时间段内，故障传感器的数据标记为未知
        if fault_start_frame <= start_frame <= fault_end_frame and lane_id in faulty_lanes:
            is_observed = 0
        else:
            is_observed = 1
        mask_data.append({
            'start_frame': start_frame,
            'lane_id': lane_id,
            'is_observed': is_observed
        })
```

### 场景4：时变观测模式

不同时间段的观测模式不同。

```python
import numpy as np

# 早高峰：主干道数据完整，支路数据稀疏
# 平峰：所有道路数据稀疏
# 晚高峰：主干道数据完整，支路数据稀疏

mask_data = []
for start_frame in all_start_frames:
    # 假设每帧代表1秒，计算小时
    hour = (start_frame // 3600) % 24
    
    for lane_id in all_lane_ids:
        is_main_road = lane_id <= 2  # 假设lane_id 1,2是主干道
        
        # 早高峰 (7-9点) 和晚高峰 (17-19点)
        if hour in [7, 8, 17, 18]:
            if is_main_road:
                is_observed = 1  # 主干道数据完整
            else:
                is_observed = 1 if np.random.rand() < 0.3 else 0  # 支路数据稀疏
        else:
            # 平峰
            is_observed = 1 if np.random.rand() < 0.5 else 0  # 所有道路数据稀疏
        
        mask_data.append({
            'start_frame': start_frame,
            'lane_id': lane_id,
            'is_observed': is_observed
        })
```

## 掩码验证

数据集加载后，可以检查掩码的统计信息：

```python
dataset = LaneTrafficDataset(
    static_data_path='data/static_road_data.csv',
    dynamic_data_path='data/dynamic_traffic_data.csv',
    mask_data_path='data/mask.csv'
)

# 检查掩码统计
print(f"已观测数据比例: {dataset.training_mask.mean():.3f}")
print(f"未观测数据比例: {dataset.eval_mask.mean():.3f}")

# 查看掩码形状
print(f"掩码形状: {dataset.training_mask.shape}")
# 输出: (n_times, n_spaces, n_features)

# 查看某个时间步的掩码
time_idx = 0
print(f"第{time_idx}个时间步的已观测节点数: {dataset.training_mask[time_idx].any(axis=1).sum()}")
```

## 注意事项

1. **时间窗口对齐**：CSV掩码文件中的start_frame必须与动态数据中的时间窗口起始帧对齐
2. **路段ID对齐**：CSV掩码文件中的lane_id必须在静态数据中存在
3. **形状匹配**：NPZ和PKL格式的掩码矩阵形状必须与数据矩阵形状匹配
4. **默认行为**：CSV格式中，如果某个(start_frame, lane_id)组合未在掩码文件中出现，默认标记为未观测（0）
5. **特征共享**：对于二维掩码，所有特征（speed, spacing等）共享相同的掩码；对于三维掩码，可以为不同特征设置不同的掩码

## 完整示例

参见 `examples/user_defined_mask_example.py` 文件，其中包含多个完整的示例代码。

## 常见问题

**Q: 如果我只想指定部分时间窗口的掩码怎么办？**

A: 在CSV格式中，未指定的(start_frame, lane_id)组合将默认标记为未观测。如果需要默认标记为已观测，需要在CSV中显式列出所有组合。

**Q: NPZ格式的掩码索引顺序是什么？**

A: 掩码的索引顺序与数据集的顺序一致：
- 第1维（行）：时间，按 `dataset.start_frames` 的顺序
- 第2维（列）：空间，按 `dataset.lane_ids` 的顺序
- 第3维：特征，按 [speed, spacing] 的顺序

**Q: 可以为不同特征设置不同的掩码吗？**

A: 可以。使用三维掩码 `[n_times, n_spaces, n_features]` 即可为每个特征设置独立的掩码。

**Q: 掩码文件可以包含额外的列吗？**

A: CSV格式可以包含额外的列，只要包含必需的三列即可。NPZ和PKL格式也可以包含额外的数组或键，但只会使用 `'mask'` 或 `'training_mask'`。

