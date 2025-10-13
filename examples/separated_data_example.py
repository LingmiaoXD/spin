"""
使用分离格式数据（静态道路数据 + 动态交通数据）的示例

本示例展示如何：
1. 生成分离格式的示例数据
2. 加载和使用分离格式的数据
3. 验证数据格式
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spin.datasets.lane_data_utils import (
    LaneDataProcessor,
    create_separated_sample_dataset,
    validate_static_data,
    validate_dynamic_data,
    migrate_to_separated_format
)
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset
import pandas as pd
import numpy as np


def example_1_create_separated_data():
    """示例 1: 创建分离格式的数据"""
    print("=" * 60)
    print("示例 1: 创建分离格式的数据")
    print("=" * 60)
    
    # 创建数据处理器
    processor = LaneDataProcessor(
        spatial_resolution=10.0,  # 10米空间分辨率
        temporal_resolution=10    # 10秒时间分辨率
    )
    
    # 创建静态道路数据
    print("\n1. 创建静态道路数据...")
    static_data = processor.create_static_road_data(
        n_lanes=3,
        lane_length=500.0,
        seed=42
    )
    
    print(f"\n静态道路数据预览：")
    print(static_data.head())
    print(f"\n数据形状: {static_data.shape}")
    print(f"列名: {static_data.columns.tolist()}")
    
    # 创建动态交通数据
    print("\n2. 创建动态交通数据...")
    dynamic_data = processor.create_dynamic_traffic_data(
        static_data=static_data,
        time_hours=1.0,  # 1小时数据
        seed=42
    )
    
    print(f"\n动态交通数据预览：")
    print(dynamic_data.head())
    print(f"\n数据形状: {dynamic_data.shape}")
    print(f"列名: {dynamic_data.columns.tolist()}")
    
    # 保存数据
    print("\n3. 保存数据...")
    processor.save_data(static_data, "data/static_road_data.csv", format='csv')
    processor.save_data(dynamic_data, "data/dynamic_traffic_data.csv", format='csv')
    
    print("\n✅ 数据创建完成！")
    return static_data, dynamic_data


def example_2_load_separated_data():
    """示例 2: 加载分离格式的数据"""
    print("\n" + "=" * 60)
    print("示例 2: 加载分离格式的数据")
    print("=" * 60)
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = LaneTrafficDataset(
        static_data_path="data/static_road_data.csv",
        dynamic_data_path="data/dynamic_traffic_data.csv",
        window_size=12,
        stride=1,
        impute_nans=True
    )
    
    # 查看数据集信息
    print(f"\n数据集信息：")
    print(f"  节点数量: {dataset.n_nodes}")
    print(f"  特征通道数: {dataset.n_channels}")
    print(f"  时间序列长度: {dataset.length}")
    print(f"  数据矩阵形状: {dataset.data.shape}")
    
    # 获取图连接矩阵
    print(f"\n图连接矩阵：")
    adj = dataset.get_connectivity()
    print(f"  邻接矩阵形状: {adj.shape}")
    print(f"  连接数: {np.sum(adj > 0) // 2}")
    
    # 获取数据
    print(f"\n获取数据：")
    data = dataset.numpy()
    print(f"  数据形状: {data.shape}")
    print(f"  速度范围: [{data[:,:,0].min():.2f}, {data[:,:,0].max():.2f}]")
    print(f"  间距范围: [{data[:,:,1].min():.2f}, {data[:,:,1].max():.2f}]")
    
    print("\n✅ 数据加载完成！")
    return dataset


def example_3_validate_data():
    """示例 3: 验证数据格式"""
    print("\n" + "=" * 60)
    print("示例 3: 验证数据格式")
    print("=" * 60)
    
    # 读取数据
    print("\n读取数据...")
    static_data = pd.read_csv("data/static_road_data.csv")
    dynamic_data = pd.read_csv("data/dynamic_traffic_data.csv")
    
    # 验证静态数据
    print("\n1. 验证静态道路数据...")
    static_valid, static_errors = validate_static_data(static_data)
    if static_valid:
        print("   ✅ 静态数据验证通过")
    else:
        print("   ❌ 静态数据验证失败:")
        for error in static_errors:
            print(f"      - {error}")
    
    # 验证动态数据
    print("\n2. 验证动态交通数据...")
    dynamic_valid, dynamic_errors = validate_dynamic_data(dynamic_data, static_data)
    if dynamic_valid:
        print("   ✅ 动态数据验证通过")
    else:
        print("   ❌ 动态数据验证失败:")
        for error in dynamic_errors:
            print(f"      - {error}")
    
    # 数据统计
    print("\n3. 数据统计信息:")
    print(f"   静态数据:")
    print(f"      节点数: {len(static_data)}")
    print(f"      车道数: {static_data['lane_id'].nunique()}")
    print(f"      节点连接规则覆盖率: {static_data['node_connections'].notna().mean():.1%}")
    
    print(f"\n   动态数据:")
    print(f"      记录数: {len(dynamic_data)}")
    print(f"      时间步数: {dynamic_data['timestamp'].nunique()}")
    print(f"      节点覆盖数: {dynamic_data['spatial_id'].nunique()}")
    print(f"      缺失值比例: {dynamic_data.isnull().mean().mean():.1%}")
    
    print("\n✅ 数据验证完成！")


def example_4_migrate_from_old_format():
    """示例 4: 从旧格式迁移到新格式"""
    print("\n" + "=" * 60)
    print("示例 4: 从旧格式迁移到新格式")
    print("=" * 60)
    
    # 创建旧格式的混合数据
    print("\n1. 创建旧格式数据（仅用于演示）...")
    processor = LaneDataProcessor()
    mixed_data = processor.create_sample_data(
        n_lanes=2,
        lane_length=300.0,
        time_hours=0.5,
        seed=42
    )
    print(f"   混合数据形状: {mixed_data.shape}")
    
    # 迁移到新格式
    print("\n2. 迁移到新格式...")
    static_data, dynamic_data = migrate_to_separated_format(
        mixed_data,
        static_cols=['lane_id', 'spatial_id', 'node_connections'],
        dynamic_cols=['timestamp', 'spatial_id', 'speed', 'spacing']
    )
    
    # 保存迁移后的数据
    print("\n3. 保存迁移后的数据...")
    processor.save_data(static_data, "data/migrated_static_data.csv")
    processor.save_data(dynamic_data, "data/migrated_dynamic_data.csv")
    
    print("\n✅ 数据迁移完成！")


def example_5_custom_connection_rules():
    """示例 5: 自定义节点连接规则"""
    print("\n" + "=" * 60)
    print("示例 5: 自定义节点连接规则")
    print("=" * 60)
    
    # 创建自定义连接规则
    print("\n1. 定义自定义节点连接规则...")
    custom_static_data = pd.DataFrame([
        {
            'lane_id': 'lane_0',
            'spatial_id': 'lane_0_0000',
            'node_connections': 'lane_0_0001,direct;lane_1_0000,dashed'
        },
        {
            'lane_id': 'lane_0',
            'spatial_id': 'lane_0_0001',
            'node_connections': 'lane_0_0000,direct;lane_0_0002,direct;lane_1_0001,dashed'
        },
        {
            'lane_id': 'lane_0',
            'spatial_id': 'lane_0_0002',
            'node_connections': 'lane_0_0001,direct;lane_1_0002,dashed'
        },
        {
            'lane_id': 'lane_1',
            'spatial_id': 'lane_1_0000',
            'node_connections': 'lane_1_0001,direct;lane_0_0000,dashed'
        },
        {
            'lane_id': 'lane_1',
            'spatial_id': 'lane_1_0001',
            'node_connections': 'lane_1_0000,direct;lane_1_0002,direct;lane_0_0001,dashed'
        },
        {
            'lane_id': 'lane_1',
            'spatial_id': 'lane_1_0002',
            'node_connections': 'lane_1_0001,direct;lane_0_0002,dashed'
        }
    ])
    
    print("   静态道路数据:")
    print(custom_static_data)
    
    # 创建对应的动态数据
    print("\n2. 创建对应的动态交通数据...")
    timestamps = pd.date_range('2024-01-01', periods=10, freq='10s')
    dynamic_records = []
    
    for timestamp in timestamps:
        for _, row in custom_static_data.iterrows():
            dynamic_records.append({
                'timestamp': timestamp,
                'spatial_id': row['spatial_id'],
                'speed': np.random.uniform(20, 40),
                'spacing': np.random.uniform(15, 30)
            })
    
    custom_dynamic_data = pd.DataFrame(dynamic_records)
    print(f"   动态数据形状: {custom_dynamic_data.shape}")
    
    # 保存数据
    print("\n3. 保存自定义数据...")
    custom_static_data.to_csv("data/custom_static_data.csv", index=False)
    custom_dynamic_data.to_csv("data/custom_dynamic_data.csv", index=False)
    
    # 加载并使用
    print("\n4. 加载自定义数据...")
    dataset = LaneTrafficDataset(
        static_data_path="data/custom_static_data.csv",
        dynamic_data_path="data/custom_dynamic_data.csv"
    )
    
    adj = dataset.get_connectivity()
    print(f"\n   图连接信息:")
    print(f"      节点数: {adj.shape[0]}")
    print(f"      连接数: {np.sum(adj > 0) // 2}")
    print(f"\n   邻接矩阵:")
    print(adj)
    
    print("\n✅ 自定义连接规则完成！")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("分离格式数据使用示例")
    print("=" * 60)
    
    # 创建数据目录
    Path("data").mkdir(exist_ok=True)
    
    try:
        # 运行示例
        static_data, dynamic_data = example_1_create_separated_data()
        dataset = example_2_load_separated_data()
        example_3_validate_data()
        example_4_migrate_from_old_format()
        example_5_custom_connection_rules()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

