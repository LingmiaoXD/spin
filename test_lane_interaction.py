"""
车道交互规则测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spin.datasets.lane_data_utils import LaneDataProcessor
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset


def test_lane_interaction():
    """测试车道交互功能"""
    print("🧪 测试车道交互规则功能...")
    
    try:
        # 1. 创建数据处理器
        processor = LaneDataProcessor(
            spatial_resolution=10.0,
            temporal_resolution=10,
            lane_interaction_col='lane_interaction'
        )
        
        # 2. 定义交互规则
        interaction_zones = [
            {
                'lanes': ['lane_0', 'lane_1'],
                'start_pos': 0,
                'end_pos': 200,
                'type': 'dashed'
            },
            {
                'lanes': ['lane_0', 'lane_1'],
                'start_pos': 200,
                'end_pos': 400,
                'type': 'solid'
            }
        ]
        
        lane_rules = processor.create_lane_interaction_rules(
            lane_ids=['lane_0', 'lane_1'],
            interaction_zones=interaction_zones
        )
        
        print("✅ 交互规则创建成功")
        
        # 3. 创建测试数据
        test_data = processor.create_sample_data(
            n_lanes=2,
            lane_length=400.0,
            time_hours=0.5,
            seed=42
        )
        
        # 手动添加交互规则
        for idx, row in test_data.iterrows():
            spatial_pos = row['spatial_position']
            if 0 <= spatial_pos <= 200:
                test_data.loc[idx, 'lane_interaction'] = 'dashed'
            else:
                test_data.loc[idx, 'lane_interaction'] = 'solid'
        
        print(f"✅ 测试数据创建成功，形状: {test_data.shape}")
        
        # 4. 测试数据集加载
        dataset = LaneTrafficDataset(
            data_path=None,  # 直接使用DataFrame
            df=test_data,
            impute_nans=True,
            lane_interaction_col='lane_interaction'
        )
        
        print("✅ 数据集加载成功")
        
        # 5. 测试图连接
        adj = dataset.get_connectivity()
        print(f"✅ 图连接构建成功")
        print(f"   - 邻接矩阵形状: {adj.shape}")
        print(f"   - 总连接数: {adj.sum()}")
        
        # 6. 分析连接模式
        print("\n📊 连接模式分析:")
        
        # 统计车道内连接
        lane_0_nodes = [i for i, spatial_id in enumerate(dataset.spatial_ids) 
                       if spatial_id.startswith('lane_0')]
        lane_1_nodes = [i for i, spatial_id in enumerate(dataset.spatial_ids) 
                       if spatial_id.startswith('lane_1')]
        
        # 车道内连接
        lane_0_internal = sum(adj[i, j] for i in lane_0_nodes for j in lane_0_nodes if i != j)
        lane_1_internal = sum(adj[i, j] for i in lane_1_nodes for j in lane_1_nodes if i != j)
        
        # 车道间连接
        cross_lane = sum(adj[i, j] for i in lane_0_nodes for j in lane_1_nodes)
        
        print(f"   - lane_0 内部连接: {lane_0_internal}")
        print(f"   - lane_1 内部连接: {lane_1_internal}")
        print(f"   - 跨车道连接: {cross_lane}")
        
        # 7. 验证交互规则
        print("\n🔍 交互规则验证:")
        
        # 检查虚线区域的连接
        dashed_connections = 0
        solid_connections = 0
        
        for i in lane_0_nodes:
            for j in lane_1_nodes:
                if adj[i, j] > 0:
                    # 获取空间位置
                    pos_i = test_data[test_data['spatial_id'] == dataset.spatial_ids[i]]['spatial_position'].iloc[0]
                    pos_j = test_data[test_data['spatial_id'] == dataset.spatial_ids[j]]['spatial_position'].iloc[0]
                    
                    # 检查是否在虚线区域
                    if 0 <= pos_i <= 200 and 0 <= pos_j <= 200:
                        dashed_connections += 1
                    else:
                        solid_connections += 1
        
        print(f"   - 虚线区域连接: {dashed_connections}")
        print(f"   - 实线区域连接: {solid_connections}")
        
        if dashed_connections > 0 and solid_connections == 0:
            print("✅ 交互规则验证通过：虚线区域有连接，实线区域无连接")
        else:
            print("❌ 交互规则验证失败")
            return False
        
        print("\n🎉 车道交互功能测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_lane_interaction()
    if success:
        print("\n✅ 车道交互规则实现成功！")
        print("\n使用方法:")
        print("1. 在数据中添加 'lane_interaction' 列")
        print("2. 使用 'dashed' 表示可交互，'solid' 表示不可交互")
        print("3. 运行训练时指定交互规则列名")
    else:
        print("\n❌ 测试失败，请检查实现")

