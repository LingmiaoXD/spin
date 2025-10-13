"""
车道级交通数据集测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spin.datasets.lane_data_utils import create_sample_dataset
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset


def test_lane_dataset():
    """测试车道数据集功能"""
    print("🧪 测试车道级交通数据集...")
    
    try:
        # 1. 创建示例数据
        print("\n1. 创建示例数据...")
        sample_data = create_sample_dataset("test_sample_data.csv")
        print(f"✅ 示例数据创建成功，形状: {sample_data.shape}")
        
        # 2. 加载数据集
        print("\n2. 加载数据集...")
        dataset = LaneTrafficDataset(
            data_path="test_sample_data.csv",
            impute_nans=True,
            window_size=12,
            stride=1
        )
        print(f"✅ 数据集加载成功")
        print(f"   - 时间步数: {dataset.length}")
        print(f"   - 空间节点数: {dataset.n_nodes}")
        print(f"   - 特征通道数: {dataset.n_channels}")
        
        # 3. 测试数据访问
        print("\n3. 测试数据访问...")
        data = dataset.numpy()
        print(f"✅ 数据访问成功，形状: {data.shape}")
        
        # 4. 测试图连接
        print("\n4. 测试图连接...")
        adj = dataset.get_connectivity()
        print(f"✅ 图连接成功，形状: {adj.shape}")
        print(f"   - 连接数: {adj.sum()}")
        
        # 5. 测试时间编码
        print("\n5. 测试时间编码...")
        time_encoding = dataset.datetime_encoded(['day', 'week'])
        print(f"✅ 时间编码成功，形状: {time_encoding.shape}")
        
        # 6. 测试数据分割
        print("\n6. 测试数据分割...")
        splitter = dataset.get_splitter()
        train_idx, val_idx, test_idx = splitter.split(dataset.length)
        print(f"✅ 数据分割成功")
        print(f"   - 训练集: {len(train_idx)}")
        print(f"   - 验证集: {len(val_idx)}")
        print(f"   - 测试集: {len(test_idx)}")
        
        print("\n🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_lane_dataset()
    if success:
        print("\n✅ 车道级交通数据集实现成功！")
        print("\n使用方法:")
        print("python experiments/run_imputation.py \\")
        print("    --model-name spin \\")
        print("    --dataset-name lane_traffic \\")
        print("    --data-path test_sample_data.csv \\")
        print("    --config config/imputation/spin_lane.yaml")
    else:
        print("\n❌ 测试失败，请检查实现")

