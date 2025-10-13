"""
车道级交通数据使用示例
演示如何使用自定义的车道级交通数据集进行数据插补
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spin.datasets.lane_data_utils import LaneDataProcessor, create_sample_dataset
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset


def main():
    """主函数：演示车道级交通数据集的完整使用流程"""
    
    print("🚗 车道级交通数据集使用示例")
    print("=" * 50)
    
    # 1. 创建示例数据
    print("\n1. 创建示例车道数据...")
    sample_data = create_sample_dataset("sample_lane_data.csv")
    print(f"✅ 示例数据创建完成，形状: {sample_data.shape}")
    
    # 2. 加载和处理数据
    print("\n2. 加载车道数据集...")
    dataset = LaneTrafficDataset(
        data_path="sample_lane_data.csv",
        impute_nans=True,
        window_size=12,
        stride=1
    )
    print(f"✅ 数据集加载完成")
    print(f"   - 时间步数: {dataset.length}")
    print(f"   - 空间节点数: {dataset.n_nodes}")
    print(f"   - 特征通道数: {dataset.n_channels}")
    
    # 3. 查看数据统计信息
    print("\n3. 数据统计信息...")
    data = dataset.numpy()
    print(f"   - 数据形状: {data.shape}")
    print(f"   - 速度范围: [{data[:, :, 0].min():.2f}, {data[:, :, 0].max():.2f}] km/h")
    print(f"   - 间距范围: [{data[:, :, 1].min():.2f}, {data[:, :, 1].max():.2f}] m")
    print(f"   - 缺失值比例: {np.isnan(data).mean():.3f}")
    
    # 4. 查看图连接信息
    print("\n4. 图连接信息...")
    adj = dataset.get_connectivity()
    print(f"   - 邻接矩阵形状: {adj.shape}")
    print(f"   - 连接数: {np.sum(adj > 0)}")
    print(f"   - 连接密度: {np.sum(adj > 0) / (adj.shape[0] * adj.shape[1]):.3f}")
    
    # 5. 查看时间编码
    print("\n5. 时间编码信息...")
    time_encoding = dataset.datetime_encoded(['day', 'week', 'hour'])
    print(f"   - 时间编码形状: {time_encoding.shape}")
    print(f"   - 时间范围: {dataset.timestamps[0]} 到 {dataset.timestamps[-1]}")
    
    # 6. 数据分割
    print("\n6. 数据分割...")
    splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)
    train_idx, val_idx, test_idx = splitter.split(dataset.length)
    print(f"   - 训练集: {len(train_idx)} 个时间步")
    print(f"   - 验证集: {len(val_idx)} 个时间步")
    print(f"   - 测试集: {len(test_idx)} 个时间步")
    
    # 7. 保存预处理数据
    print("\n7. 保存预处理数据...")
    dataset.save_processed_data("processed_lane_data.pkl")
    print("✅ 预处理数据已保存")
    
    # 8. 演示如何运行训练
    print("\n8. 运行训练命令示例...")
    print("使用以下命令运行车道级交通数据插补训练：")
    print("python experiments/run_imputation.py \\")
    print("    --model-name spin \\")
    print("    --dataset-name lane_traffic \\")
    print("    --data-path sample_lane_data.csv \\")
    print("    --config config/imputation/spin_lane.yaml \\")
    print("    --epochs 100 \\")
    print("    --batch-size 8")
    
    print("\n🎉 示例完成！")


def create_custom_lane_data():
    """创建自定义车道数据的示例"""
    print("\n📝 创建自定义车道数据示例")
    print("-" * 30)
    
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
    
    # 创建车道信息
    lane_info = processor.create_lane_info(
        lane_ids=['lane_0', 'lane_1', 'lane_2'],
        lane_lengths=[1000.0, 1200.0, 800.0],
        lane_positions=[(0, 0), (0, 3.5), (0, 7.0)]  # 车道位置坐标
    )
    
    print("车道信息:")
    for lane_id, info in lane_info.items():
        print(f"  {lane_id}: 长度={info['length']}m, 位置={info['position']}")
    
    # 创建示例数据
    sample_data = processor.create_sample_data(
        n_lanes=3,
        lane_length=1000.0,
        time_hours=1.0,  # 1小时数据
        seed=123
    )
    
    # 验证数据
    if processor.validate_data(sample_data):
        print("✅ 自定义数据验证通过")
        
        # 保存数据
        processor.save_data(sample_data, "custom_lane_data.csv", format='csv')
        print("✅ 自定义数据已保存到 custom_lane_data.csv")
    else:
        print("❌ 自定义数据验证失败")


if __name__ == "__main__":
    # 运行主示例
    main()
    
    # 运行自定义数据创建示例
    create_custom_lane_data()

