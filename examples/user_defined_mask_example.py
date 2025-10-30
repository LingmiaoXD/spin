"""
用户自定义掩码示例

演示如何使用用户自定义掩码来指定哪些spatial_id在哪些时间戳下的值是已知的
"""

import numpy as np
import pandas as pd
from pathlib import Path

def create_complex_mask_scenario():
    """
    创建更复杂的掩码场景
    
    场景：在不同时间段，不同spatial_id的观测情况不同
    - 早上时段：只有主干道的数据是已知的
    - 中午时段：所有道路的数据都是已知的
    - 晚上时段：只有部分道路的数据是已知的
    """
    timestamps = pd.date_range('2024-01-01 06:00:00', periods=180, freq='10S')  # 30分钟数据
    spatial_ids = [
        'lane_0_0000', 'lane_0_0001', 'lane_0_0002',  # 主干道
        'lane_1_0000', 'lane_1_0001',  # 支路1
        'lane_2_0000', 'lane_2_0001'   # 支路2
    ]
    
    mask_data = []
    
    for i, timestamp in enumerate(timestamps):
        for spatial_id in spatial_ids:
            # 根据时间段和道路类型决定是否已观测
            if i < 60:  # 早上时段 (前10分钟)
                # 只有主干道是已知的
                is_observed = spatial_id.startswith('lane_0')
            elif i < 120:  # 中午时段 (中间10分钟)
                # 所有道路都是已知的
                is_observed = True
            else:  # 晚上时段 (后10分钟)
                # 主干道和支路1是已知的
                is_observed = spatial_id.startswith('lane_0') or spatial_id.startswith('lane_1')
            
            mask_data.append({
                'timestamp': timestamp,
                'spatial_id': spatial_id,
                'is_observed': is_observed
            })
    
    mask_df = pd.DataFrame(mask_data)
    mask_df.to_csv('mask_complex_example.csv', index=False)
    
    print(f"✅ 复杂掩码场景已创建: mask_complex_example.csv")
    print(f"   总记录数: {len(mask_df)}")
    print(f"   已观测记录数: {mask_df['is_observed'].sum()}")
    print(f"   已观测比例: {mask_df['is_observed'].mean():.3f}")
    
    # 统计各时间段的观测比例
    mask_df['time_period'] = pd.cut(mask_df.index, bins=[0, 420, 840, 1260], labels=['早上', '中午', '晚上'])
    print("\n各时间段观测比例:")
    print(mask_df.groupby('time_period')['is_observed'].mean())
    
    return mask_df


if __name__ == '__main__':
    create_complex_mask_scenario()
    
    print("\n" + "=" * 60)
    print("掩码文件已创建完成！")
    print("=" * 60)


