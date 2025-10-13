"""
测试用户自定义掩码功能
"""

import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


def create_test_data():
    """创建测试用的静态和动态数据"""
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    # 创建静态道路数据
    static_data = {
        'lane_id': ['lane_0', 'lane_0', 'lane_1', 'lane_1'],
        'spatial_id': ['lane_0_0000', 'lane_0_0001', 'lane_1_0000', 'lane_1_0001'],
        'node_connections': [
            'lane_1_0000,dashed',
            'lane_1_0001,dashed',
            'lane_0_0000,dashed',
            'lane_0_0001,dashed'
        ]
    }
    static_df = pd.DataFrame(static_data)
    static_path = os.path.join(temp_dir, 'static_data.csv')
    static_df.to_csv(static_path, index=False)
    
    # 创建动态交通数据
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=10, freq='10S')
    spatial_ids = ['lane_0_0000', 'lane_0_0001', 'lane_1_0000', 'lane_1_0001']
    
    dynamic_data = []
    for timestamp in timestamps:
        for spatial_id in spatial_ids:
            dynamic_data.append({
                'timestamp': timestamp,
                'spatial_id': spatial_id,
                'speed': np.random.uniform(20, 60),
                'spacing': np.random.uniform(5, 15)
            })
    
    dynamic_df = pd.DataFrame(dynamic_data)
    dynamic_path = os.path.join(temp_dir, 'dynamic_data.csv')
    dynamic_df.to_csv(dynamic_path, index=False)
    
    return temp_dir, static_path, dynamic_path, timestamps, spatial_ids


def test_csv_mask():
    """测试CSV格式的用户自定义掩码"""
    print("\n" + "="*60)
    print("测试 CSV 格式掩码")
    print("="*60)
    
    temp_dir, static_path, dynamic_path, timestamps, spatial_ids = create_test_data()
    
    # 创建CSV格式的掩码
    mask_data = []
    for timestamp in timestamps:
        for spatial_id in spatial_ids:
            # 只有前两个spatial_id是已知的
            is_observed = spatial_id in ['lane_0_0000', 'lane_1_0000']
            mask_data.append({
                'timestamp': timestamp,
                'spatial_id': spatial_id,
                'is_observed': is_observed
            })
    
    mask_df = pd.DataFrame(mask_data)
    mask_path = os.path.join(temp_dir, 'mask.csv')
    mask_df.to_csv(mask_path, index=False)
    
    print(f"创建的掩码文件:")
    print(f"  已观测记录数: {mask_df['is_observed'].sum()}")
    print(f"  未观测记录数: {(~mask_df['is_observed']).sum()}")
    print(f"  已观测比例: {mask_df['is_observed'].mean():.3f}")
    
    # 加载数据集
    from spin.datasets.lane_traffic_dataset import LaneTrafficDataset
    
    dataset = LaneTrafficDataset(
        static_data_path=static_path,
        dynamic_data_path=dynamic_path,
        mask_data_path=mask_path,
        time_col='timestamp',
        spatial_id_col='spatial_id',
        speed_col='speed',
        spacing_col='spacing'
    )
    
    print(f"\n数据集加载成功:")
    print(f"  数据形状: {dataset.data.shape}")
    print(f"  掩码形状: {dataset.training_mask.shape}")
    print(f"  已观测数据比例: {dataset.training_mask.mean():.3f}")
    print(f"  未观测数据比例: {dataset.eval_mask.mean():.3f}")
    
    # 验证掩码是否正确
    expected_observed_ratio = 0.5  # 4个spatial_id中有2个是已知的
    actual_observed_ratio = dataset.training_mask.mean()
    
    if abs(expected_observed_ratio - actual_observed_ratio) < 0.01:
        print(f"  ✅ 掩码验证通过")
    else:
        print(f"  ❌ 掩码验证失败: 期望 {expected_observed_ratio:.3f}, 实际 {actual_observed_ratio:.3f}")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    return True


def test_npz_mask():
    """测试NPZ格式的用户自定义掩码"""
    print("\n" + "="*60)
    print("测试 NPZ 格式掩码")
    print("="*60)
    
    temp_dir, static_path, dynamic_path, timestamps, spatial_ids = create_test_data()
    
    # 创建NPZ格式的掩码
    n_times = len(timestamps)
    n_spaces = len(spatial_ids)
    
    # 前2个spatial_id是已知的
    mask = np.zeros((n_times, n_spaces), dtype=bool)
    mask[:, :2] = True
    
    mask_path = os.path.join(temp_dir, 'mask.npz')
    np.savez(mask_path, mask=mask)
    
    print(f"创建的掩码文件:")
    print(f"  掩码形状: {mask.shape}")
    print(f"  已观测比例: {mask.mean():.3f}")
    
    # 加载数据集
    from spin.datasets.lane_traffic_dataset import LaneTrafficDataset
    
    dataset = LaneTrafficDataset(
        static_data_path=static_path,
        dynamic_data_path=dynamic_path,
        mask_data_path=mask_path,
        time_col='timestamp',
        spatial_id_col='spatial_id',
        speed_col='speed',
        spacing_col='spacing'
    )
    
    print(f"\n数据集加载成功:")
    print(f"  数据形状: {dataset.data.shape}")
    print(f"  掩码形状: {dataset.training_mask.shape}")
    print(f"  已观测数据比例: {dataset.training_mask.mean():.3f}")
    print(f"  未观测数据比例: {dataset.eval_mask.mean():.3f}")
    
    # 验证掩码是否正确
    expected_observed_ratio = 0.5
    actual_observed_ratio = dataset.training_mask.mean()
    
    if abs(expected_observed_ratio - actual_observed_ratio) < 0.01:
        print(f"  ✅ 掩码验证通过")
    else:
        print(f"  ❌ 掩码验证失败: 期望 {expected_observed_ratio:.3f}, 实际 {actual_observed_ratio:.3f}")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    return True


def test_pkl_mask():
    """测试PKL格式的用户自定义掩码"""
    print("\n" + "="*60)
    print("测试 PKL 格式掩码")
    print("="*60)
    
    temp_dir, static_path, dynamic_path, timestamps, spatial_ids = create_test_data()
    
    # 创建PKL格式的掩码
    import pickle
    
    n_times = len(timestamps)
    n_spaces = len(spatial_ids)
    n_features = 2
    
    # 前2个spatial_id是已知的
    training_mask = np.zeros((n_times, n_spaces, n_features), dtype=bool)
    training_mask[:, :2, :] = True
    
    mask_data = {
        'training_mask': training_mask
    }
    
    mask_path = os.path.join(temp_dir, 'mask.pkl')
    with open(mask_path, 'wb') as f:
        pickle.dump(mask_data, f)
    
    print(f"创建的掩码文件:")
    print(f"  掩码形状: {training_mask.shape}")
    print(f"  已观测比例: {training_mask.mean():.3f}")
    
    # 加载数据集
    from spin.datasets.lane_traffic_dataset import LaneTrafficDataset
    
    dataset = LaneTrafficDataset(
        static_data_path=static_path,
        dynamic_data_path=dynamic_path,
        mask_data_path=mask_path,
        time_col='timestamp',
        spatial_id_col='spatial_id',
        speed_col='speed',
        spacing_col='spacing'
    )
    
    print(f"\n数据集加载成功:")
    print(f"  数据形状: {dataset.data.shape}")
    print(f"  掩码形状: {dataset.training_mask.shape}")
    print(f"  已观测数据比例: {dataset.training_mask.mean():.3f}")
    print(f"  未观测数据比例: {dataset.eval_mask.mean():.3f}")
    
    # 验证掩码是否正确
    expected_observed_ratio = 0.5
    actual_observed_ratio = dataset.training_mask.mean()
    
    if abs(expected_observed_ratio - actual_observed_ratio) < 0.01:
        print(f"  ✅ 掩码验证通过")
    else:
        print(f"  ❌ 掩码验证失败: 期望 {expected_observed_ratio:.3f}, 实际 {actual_observed_ratio:.3f}")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    return True


def test_default_mask():
    """测试默认的随机掩码"""
    print("\n" + "="*60)
    print("测试默认随机掩码（不提供mask_data_path）")
    print("="*60)
    
    temp_dir, static_path, dynamic_path, timestamps, spatial_ids = create_test_data()
    
    # 加载数据集（不提供掩码文件）
    from spin.datasets.lane_traffic_dataset import LaneTrafficDataset
    
    dataset = LaneTrafficDataset(
        static_data_path=static_path,
        dynamic_data_path=dynamic_path,
        # mask_data_path=None,  # 不提供掩码文件
        time_col='timestamp',
        spatial_id_col='spatial_id',
        speed_col='speed',
        spacing_col='spacing'
    )
    
    print(f"\n数据集加载成功:")
    print(f"  数据形状: {dataset.data.shape}")
    print(f"  掩码形状: {dataset.training_mask.shape}")
    print(f"  已观测数据比例: {dataset.training_mask.mean():.3f}")
    print(f"  未观测数据比例: {dataset.eval_mask.mean():.3f}")
    
    # 验证是否使用了随机掩码（默认20%未观测）
    if 0.7 < dataset.training_mask.mean() < 0.9:
        print(f"  ✅ 使用了默认随机掩码")
    else:
        print(f"  ⚠️  掩码比例异常")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    return True


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# 用户自定义掩码功能测试")
    print("#"*60)
    
    try:
        # 测试各种格式的掩码
        test_csv_mask()
        test_npz_mask()
        test_pkl_mask()
        test_default_mask()
        
        print("\n" + "#"*60)
        print("# ✅ 所有测试通过！")
        print("#"*60 + "\n")
        
    except Exception as e:
        print("\n" + "#"*60)
        print(f"# ❌ 测试失败: {e}")
        print("#"*60 + "\n")
        import traceback
        traceback.print_exc()

