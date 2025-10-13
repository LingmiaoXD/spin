"""
车道交互规则使用示例
演示如何设置车道间的交互规则（虚线可交互，实线不可交互）
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spin.datasets.lane_data_utils import LaneDataProcessor
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset


def create_lane_interaction_example():
    """创建车道交互规则示例"""
    print("🚗 车道交互规则示例")
    print("=" * 50)
    
    # 1. 创建数据处理器
    processor = LaneDataProcessor(
        spatial_resolution=10.0,  # 10米空间分辨率
        temporal_resolution=10,   # 10秒时间分辨率
        lane_interaction_col='lane_interaction'
    )
    
    # 2. 定义车道交互规则
    print("\n1. 定义车道交互规则...")
    
    # 定义交互区域
    interaction_zones = [
        # 区域1：0-200米，lane_0和lane_1可以交互（虚线）
        {
            'lanes': ['lane_0', 'lane_1'],
            'start_pos': 0,
            'end_pos': 200,
            'type': 'dashed'
        },
        # 区域2：200-400米，所有车道都不可以交互（实线）
        {
            'lanes': ['lane_0', 'lane_1', 'lane_2'],
            'start_pos': 200,
            'end_pos': 400,
            'type': 'solid'
        },
        # 区域3：400-600米，lane_1和lane_2可以交互（虚线）
        {
            'lanes': ['lane_1', 'lane_2'],
            'start_pos': 400,
            'end_pos': 600,
            'type': 'dashed'
        },
        # 区域4：600-800米，所有车道都不可以交互（实线）
        {
            'lanes': ['lane_0', 'lane_1', 'lane_2'],
            'start_pos': 600,
            'end_pos': 800,
            'type': 'solid'
        },
        # 区域5：800-1000米，lane_0和lane_2可以交互（虚线）
        {
            'lanes': ['lane_0', 'lane_2'],
            'start_pos': 800,
            'end_pos': 1000,
            'type': 'dashed'
        }
    ]
    
    # 创建车道交互规则
    lane_rules = processor.create_lane_interaction_rules(
        lane_ids=['lane_0', 'lane_1', 'lane_2'],
        interaction_zones=interaction_zones
    )
    
    print("✅ 车道交互规则创建完成")
    for lane_id, rules in lane_rules.items():
        print(f"   {lane_id}: {len(rules['interaction_rules'])} 个交互区域")
        for rule in rules['interaction_rules']:
            start_pos, end_pos, interaction_type = rule
            print(f"     - {start_pos}-{end_pos}m: {interaction_type}")
    
    # 3. 创建车道信息
    print("\n2. 创建车道信息...")
    lane_info = processor.create_lane_info(
        lane_ids=['lane_0', 'lane_1', 'lane_2'],
        lane_lengths=[1000.0, 1000.0, 1000.0],
        lane_positions=[(0, 0), (0, 3.5), (0, 7.0)]  # 车道位置坐标
    )
    
    # 将交互规则添加到车道信息中
    for lane_id in lane_info:
        lane_info[lane_id].update(lane_rules[lane_id])
    
    print("✅ 车道信息创建完成")
    
    # 4. 创建示例数据
    print("\n3. 创建示例数据...")
    sample_data = processor.create_sample_data(
        n_lanes=3,
        lane_length=1000.0,
        time_hours=1.0,  # 1小时数据
        seed=123
    )
    
    # 手动添加交互规则到数据中
    for _, row in sample_data.iterrows():
        lane_id = row['lane_id']
        spatial_pos = row['spatial_position']
        
        # 根据位置确定交互规则
        interaction_type = 'solid'  # 默认实线
        for zone in interaction_zones:
            if lane_id in zone['lanes'] and zone['start_pos'] <= spatial_pos <= zone['end_pos']:
                interaction_type = zone['type']
                break
                
        sample_data.loc[sample_data.index == row.name, 'lane_interaction'] = interaction_type
    
    print(f"✅ 示例数据创建完成，形状: {sample_data.shape}")
    print(f"   交互规则分布:")
    print(f"   - 虚线(dashed): {(sample_data['lane_interaction'] == 'dashed').sum()}")
    print(f"   - 实线(solid): {(sample_data['lane_interaction'] == 'solid').sum()}")
    
    # 5. 保存数据
    print("\n4. 保存数据...")
    processor.save_data(sample_data, "lane_interaction_data.csv", format='csv')
    print("✅ 数据已保存到 lane_interaction_data.csv")
    
    return sample_data, lane_info


def test_lane_interaction_dataset():
    """测试车道交互数据集"""
    print("\n🧪 测试车道交互数据集...")
    
    try:
        # 加载数据集
        dataset = LaneTrafficDataset(
            data_path="lane_interaction_data.csv",
            impute_nans=True,
            window_size=12,
            stride=1,
            lane_interaction_col='lane_interaction'
        )
        
        print(f"✅ 数据集加载成功")
        print(f"   - 时间步数: {dataset.length}")
        print(f"   - 空间节点数: {dataset.n_nodes}")
        print(f"   - 特征通道数: {dataset.n_channels}")
        
        # 查看图连接
        adj = dataset.get_connectivity()
        print(f"✅ 图连接构建成功")
        print(f"   - 邻接矩阵形状: {adj.shape}")
        print(f"   - 总连接数: {adj.sum()}")
        
        # 分析车道间连接
        print(f"\n📊 车道间连接分析:")
        
        # 获取空间ID到车道ID的映射
        spatial_to_lane = {}
        for _, row in dataset.df.iterrows():
            spatial_to_lane[row['spatial_id']] = row['lane_id']
        
        # 统计不同车道间的连接
        lane_connections = {}
        for i in range(len(dataset.spatial_ids)):
            for j in range(i+1, len(dataset.spatial_ids)):
                if adj[i, j] > 0:
                    lane1 = spatial_to_lane[dataset.spatial_ids[i]]
                    lane2 = spatial_to_lane[dataset.spatial_ids[j]]
                    
                    if lane1 != lane2:  # 跨车道连接
                        key = tuple(sorted([lane1, lane2]))
                        if key not in lane_connections:
                            lane_connections[key] = 0
                        lane_connections[key] += 1
        
        for (lane1, lane2), count in lane_connections.items():
            print(f"   - {lane1} ↔ {lane2}: {count} 个连接")
        
        print(f"\n🎉 车道交互数据集测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def visualize_lane_interaction():
    """可视化车道交互规则"""
    print("\n📊 车道交互规则可视化...")
    
    try:
        # 读取数据
        data = pd.read_csv("lane_interaction_data.csv")
        
        # 按车道和位置分组
        lane_data = {}
        for lane_id in data['lane_id'].unique():
            lane_df = data[data['lane_id'] == lane_id].sort_values('spatial_position')
            lane_data[lane_id] = lane_df
        
        # 创建可视化
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(lane_data), 1, figsize=(12, 8))
        if len(lane_data) == 1:
            axes = [axes]
        
        colors = {'dashed': 'orange', 'solid': 'red'}
        
        for i, (lane_id, lane_df) in enumerate(lane_data.items()):
            ax = axes[i]
            
            # 绘制交互规则
            for _, row in lane_df.iterrows():
                pos = row['spatial_position']
                interaction = row['lane_interaction']
                color = colors[interaction]
                
                ax.barh(0, 10, left=pos, height=0.5, color=color, alpha=0.7)
            
            ax.set_xlim(0, 1000)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([0])
            ax.set_yticklabels([lane_id])
            ax.set_xlabel('位置 (米)')
            ax.set_title(f'{lane_id} 车道交互规则')
            
            # 添加图例
            if i == 0:
                ax.legend(['虚线(可交互)', '实线(不可交互)'], 
                         loc='upper right', bbox_to_anchor=(1, 1.2))
        
        plt.tight_layout()
        plt.savefig('lane_interaction_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可视化图表已保存为 lane_interaction_visualization.png")
        
    except ImportError:
        print("⚠️  matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"❌ 可视化失败: {str(e)}")


def main():
    """主函数"""
    print("🚗 车道交互规则完整示例")
    print("=" * 60)
    
    # 1. 创建车道交互示例数据
    sample_data, lane_info = create_lane_interaction_example()
    
    # 2. 测试数据集
    success = test_lane_interaction_dataset()
    
    # 3. 可视化（可选）
    visualize_lane_interaction()
    
    if success:
        print("\n🎉 车道交互规则示例完成！")
        print("\n使用方法:")
        print("python experiments/run_imputation.py \\")
        print("    --model-name spin \\")
        print("    --dataset-name lane_traffic \\")
        print("    --data-path lane_interaction_data.csv \\")
        print("    --config config/imputation/spin_lane.yaml")
    else:
        print("\n❌ 示例失败，请检查实现")


if __name__ == "__main__":
    main()

