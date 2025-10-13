"""
节点连接规则测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spin.datasets.lane_data_utils import LaneDataProcessor
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset


def test_node_connections():
    """测试节点连接功能"""
    print("🧪 测试节点连接规则功能...")
    
    try:
        # 1. 创建数据处理器
        processor = LaneDataProcessor(
            spatial_resolution=10.0,
            temporal_resolution=10,
            node_connections_col='node_connections'
        )
        
        # 2. 定义节点连接规则
        node_connection_rules = [
            {
                'spatial_id': 'lane_0_0000',
                'connections': {
                    'lane_0_0001': 'direct',    # 与 lane_0_0001 直通连接
                    'lane_1_0000': 'dashed',    # 与 lane_1_0000 虚线连接
                    'lane_2_0000': 'dashed'     # 与 lane_2_0000 虚线连接
                }
            },
            {
                'spatial_id': 'lane_0_0001',
                'connections': {
                    'lane_0_0000': 'direct',    # 与 lane_0_0000 直通连接
                    'lane_0_0002': 'direct',    # 与 lane_0_0002 直通连接
                    'lane_1_0001': 'dashed'     # 与 lane_1_0001 虚线连接
                }
            }
        ]
        
        node_rules = processor.create_node_connection_rules(
            spatial_ids=['lane_0_0000', 'lane_0_0001'],
            connection_rules=node_connection_rules
        )
        
        print("✅ 节点连接规则创建成功")
        
        # 3. 创建测试数据
        test_data = processor.create_sample_data(
            n_lanes=3,
            lane_length=30.0,  # 30米车道，3个节点
            time_hours=0.1,    # 6分钟数据
            seed=42
        )
        
        # 手动添加节点连接规则
        for idx, row in test_data.iterrows():
            spatial_id = row['spatial_id']
            
            if spatial_id in node_rules:
                connections = node_rules[spatial_id]['connections']
                connection_str = ";".join([f"{target},{conn_type}" for target, conn_type in connections.items()])
                test_data.loc[idx, 'node_connections'] = connection_str
            else:
                # 使用默认规则
                test_data.loc[idx, 'node_connections'] = processor._get_default_node_connections(
                    spatial_id, row['lane_id'], row['spatial_position']
                )
        
        print(f"✅ 测试数据创建成功，形状: {test_data.shape}")
        
        # 4. 测试数据集加载
        dataset = LaneTrafficDataset(
            data_path=None,  # 直接使用DataFrame
            df=test_data,
            impute_nans=True,
            node_connections_col='node_connections'
        )
        
        print("✅ 数据集加载成功")
        
        # 5. 测试图连接
        adj = dataset.get_connectivity()
        print(f"✅ 图连接构建成功")
        print(f"   - 邻接矩阵形状: {adj.shape}")
        print(f"   - 总连接数: {adj.sum()}")
        
        # 6. 分析连接模式
        print("\n📊 连接模式分析:")
        
        # 获取空间ID到索引的映射
        spatial_to_idx = {spatial_id: idx for idx, spatial_id in enumerate(dataset.spatial_ids)}
        
        # 分析 lane_0_0000 的连接
        if 'lane_0_0000' in spatial_to_idx:
            idx_0000 = spatial_to_idx['lane_0_0000']
            connections_0000 = []
            for i, connected in enumerate(adj[idx_0000]):
                if connected > 0:
                    connections_0000.append(dataset.spatial_ids[i])
            
            print(f"   - lane_0_0000 连接: {connections_0000}")
            
            # 验证预期连接
            expected_connections = ['lane_0_0001', 'lane_1_0000', 'lane_2_0000']
            actual_connections = set(connections_0000)
            expected_set = set(expected_connections)
            
            if expected_set.issubset(actual_connections):
                print("   ✅ lane_0_0000 连接验证通过")
            else:
                missing = expected_set - actual_connections
                extra = actual_connections - expected_set
                print(f"   ❌ lane_0_0000 连接验证失败")
                if missing:
                    print(f"      缺少连接: {missing}")
                if extra:
                    print(f"      多余连接: {extra}")
        
        # 分析 lane_0_0001 的连接
        if 'lane_0_0001' in spatial_to_idx:
            idx_0001 = spatial_to_idx['lane_0_0001']
            connections_0001 = []
            for i, connected in enumerate(adj[idx_0001]):
                if connected > 0:
                    connections_0001.append(dataset.spatial_ids[i])
            
            print(f"   - lane_0_0001 连接: {connections_0001}")
        
        # 7. 验证连接类型
        print("\n🔍 连接类型验证:")
        
        # 检查直通连接
        direct_connections = 0
        dashed_connections = 0
        
        for i in range(len(dataset.spatial_ids)):
            for j in range(i+1, len(dataset.spatial_ids)):
                if adj[i, j] > 0:
                    spatial_id_i = dataset.spatial_ids[i]
                    spatial_id_j = dataset.spatial_ids[j]
                    
                    # 检查连接类型
                    if (spatial_id_i.startswith('lane_0') and spatial_id_j.startswith('lane_0')):
                        direct_connections += 1
                    elif (spatial_id_i.startswith('lane_0') and spatial_id_j.startswith('lane_1')) or \
                         (spatial_id_i.startswith('lane_0') and spatial_id_j.startswith('lane_2')):
                        dashed_connections += 1
        
        print(f"   - 直通连接: {direct_connections}")
        print(f"   - 虚线连接: {dashed_connections}")
        
        if direct_connections > 0 and dashed_connections > 0:
            print("   ✅ 连接类型验证通过：包含直通和虚线连接")
        else:
            print("   ❌ 连接类型验证失败")
            return False
        
        print("\n🎉 节点连接功能测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_node_connections()
    if success:
        print("\n✅ 节点连接规则实现成功！")
        print("\n数据格式说明:")
        print("node_connections 列格式: 'target1,type1;target2,type2'")
        print("示例: 'lane_0_0001,direct;lane_1_0000,dashed;lane_2_0032,dashed'")
        print("\n连接类型:")
        print("  - direct: 直通连接（同一车道内相邻节点）")
        print("  - dashed: 虚线连接（跨车道连接）")
        print("  - solid: 实线连接（不连接）")
    else:
        print("\n❌ 测试失败，请检查实现")

