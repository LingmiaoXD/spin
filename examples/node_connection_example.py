"""
节点级连接规则使用示例
演示如何为每个节点指定具体的连接规则
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


def create_node_connection_example():
    """创建节点级连接规则示例"""
    print("🔗 节点级连接规则示例")
    print("=" * 50)
    
    # 1. 创建数据处理器
    processor = LaneDataProcessor(
        spatial_resolution=10.0,  # 10米空间分辨率
        temporal_resolution=10,   # 10秒时间分辨率
        node_connections_col='node_connections'
    )
    
    # 2. 定义节点连接规则
    print("\n1. 定义节点连接规则...")
    
    # 为特定节点定义连接规则
    node_connection_rules = [
        # lane_0_0000 的连接规则
        {
            'spatial_id': 'lane_0_0000',
            'connections': {
                'lane_0_0001': 'direct',    # 与 lane_0_0001 直通连接
                'lane_1_0000': 'dashed',    # 与 lane_1_0000 虚线连接
                'lane_2_0032': 'dashed'     # 与 lane_2_0032 虚线连接
            }
        },
        # lane_0_0001 的连接规则
        {
            'spatial_id': 'lane_0_0001',
            'connections': {
                'lane_0_0000': 'direct',    # 与 lane_0_0000 直通连接
                'lane_0_0002': 'direct',    # 与 lane_0_0002 直通连接
                'lane_1_0001': 'dashed',    # 与 lane_1_0001 虚线连接
                'lane_2_0001': 'dashed'     # 与 lane_2_0001 虚线连接
            }
        },
        # lane_1_0000 的连接规则
        {
            'spatial_id': 'lane_1_0000',
            'connections': {
                'lane_1_0001': 'direct',    # 与 lane_1_0001 直通连接
                'lane_0_0000': 'dashed',    # 与 lane_0_0000 虚线连接
                'lane_2_0000': 'dashed'     # 与 lane_2_0000 虚线连接
            }
        },
        # lane_2_0032 的连接规则
        {
            'spatial_id': 'lane_2_0032',
            'connections': {
                'lane_2_0031': 'direct',    # 与 lane_2_0031 直通连接
                'lane_2_0033': 'direct',    # 与 lane_2_0033 直通连接
                'lane_0_0000': 'dashed',    # 与 lane_0_0000 虚线连接
                'lane_1_0032': 'dashed'     # 与 lane_1_0032 虚线连接
            }
        }
    ]
    
    # 创建节点连接规则
    node_rules = processor.create_node_connection_rules(
        spatial_ids=['lane_0_0000', 'lane_0_0001', 'lane_1_0000', 'lane_2_0032'],
        connection_rules=node_connection_rules
    )
    
    print("✅ 节点连接规则创建完成")
    for spatial_id, rules in node_rules.items():
        print(f"   {spatial_id}: {len(rules['connections'])} 个连接")
        for target, conn_type in rules['connections'].items():
            print(f"     - {target}: {conn_type}")
    
    # 3. 创建车道信息
    print("\n2. 创建车道信息...")
    lane_info = processor.create_lane_info(
        lane_ids=['lane_0', 'lane_1', 'lane_2'],
        lane_lengths=[100.0, 100.0, 100.0],  # 100米车道，10个节点
        lane_positions=[(0, 0), (0, 3.5), (0, 7.0)]
    )
    
    # 将节点连接规则添加到车道信息中
    for lane_id in lane_info:
        lane_info[lane_id]['node_connections'] = {}
        for spatial_id, rules in node_rules.items():
            if spatial_id.startswith(lane_id):
                lane_info[lane_id]['node_connections'][spatial_id] = rules['connections']
    
    print("✅ 车道信息创建完成")
    
    # 4. 创建示例数据
    print("\n3. 创建示例数据...")
    sample_data = processor.create_sample_data(
        n_lanes=3,
        lane_length=100.0,  # 100米车道
        time_hours=0.5,     # 30分钟数据
        seed=123
    )
    
    # 手动添加节点连接规则到数据中
    for idx, row in sample_data.iterrows():
        spatial_id = row['spatial_id']
        
        # 根据预定义规则设置连接
        if spatial_id in node_rules:
            connections = node_rules[spatial_id]['connections']
            connection_str = ";".join([f"{target},{conn_type}" for target, conn_type in connections.items()])
            sample_data.loc[idx, 'node_connections'] = connection_str
        else:
            # 使用默认规则
            sample_data.loc[idx, 'node_connections'] = processor._get_default_node_connections(
                spatial_id, row['lane_id'], row['spatial_position']
            )
    
    print(f"✅ 示例数据创建完成，形状: {sample_data.shape}")
    print(f"   节点连接规则分布:")
    connection_types = sample_data['node_connections'].str.count('dashed')
    print(f"   - 包含虚线连接: {(connection_types > 0).sum()}")
    print(f"   - 包含直通连接: {(sample_data['node_connections'].str.count('direct') > 0).sum()}")
    
    # 5. 保存数据
    print("\n4. 保存数据...")
    processor.save_data(sample_data, "node_connection_data.csv", format='csv')
    print("✅ 数据已保存到 node_connection_data.csv")
    
    return sample_data, lane_info


def test_node_connection_dataset():
    """测试节点连接数据集"""
    print("\n🧪 测试节点连接数据集...")
    
    try:
        # 加载数据集
        dataset = LaneTrafficDataset(
            data_path="node_connection_data.csv",
            impute_nans=True,
            window_size=12,
            stride=1,
            node_connections_col='node_connections'
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
        
        # 分析特定节点的连接
        print(f"\n📊 特定节点连接分析:")
        
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
            expected_connections = ['lane_0_0001', 'lane_1_0000', 'lane_2_0032']
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
        
        # 分析 lane_1_0000 的连接
        if 'lane_1_0000' in spatial_to_idx:
            idx_1000 = spatial_to_idx['lane_1_0000']
            connections_1000 = []
            for i, connected in enumerate(adj[idx_1000]):
                if connected > 0:
                    connections_1000.append(dataset.spatial_ids[i])
            
            print(f"   - lane_1_0000 连接: {connections_1000}")
        
        # 分析 lane_2_0032 的连接
        if 'lane_2_0032' in spatial_to_idx:
            idx_2032 = spatial_to_idx['lane_2_0032']
            connections_2032 = []
            for i, connected in enumerate(adj[idx_2032]):
                if connected > 0:
                    connections_2032.append(dataset.spatial_ids[i])
            
            print(f"   - lane_2_0032 连接: {connections_2032}")
        
        print(f"\n🎉 节点连接数据集测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def visualize_node_connections():
    """可视化节点连接"""
    print("\n📊 节点连接可视化...")
    
    try:
        # 读取数据
        data = pd.read_csv("node_connection_data.csv")
        
        # 创建连接图
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for _, row in data.iterrows():
            spatial_id = row['spatial_id']
            lane_id = row['lane_id']
            G.add_node(spatial_id, lane=lane_id)
        
        # 添加边（基于连接规则）
        for _, row in data.iterrows():
            spatial_id = row['spatial_id']
            connections = row['node_connections']
            
            if pd.notna(connections):
                for connection in connections.split(';'):
                    if ',' in connection:
                        target, conn_type = connection.strip().split(',', 1)
                        if target in G.nodes():
                            G.add_edge(spatial_id, target, type=conn_type)
        
        # 绘制图
        plt.figure(figsize=(15, 10))
        
        # 按车道着色
        lane_colors = {'lane_0': 'red', 'lane_1': 'blue', 'lane_2': 'green'}
        node_colors = [lane_colors.get(G.nodes[node]['lane'], 'gray') for node in G.nodes()]
        
        # 按连接类型设置边样式
        direct_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'direct']
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'dashed']
        
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edgelist=direct_edges, edge_color='black', width=2, style='-')
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color='orange', width=2, style='--')
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # 添加图例
        plt.legend(['直通连接', '虚线连接'], loc='upper right')
        plt.title('节点连接规则可视化')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('node_connections_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可视化图表已保存为 node_connections_visualization.png")
        
    except ImportError:
        print("⚠️  matplotlib/networkx未安装，跳过可视化")
    except Exception as e:
        print(f"❌ 可视化失败: {str(e)}")


def main():
    """主函数"""
    print("🔗 节点级连接规则完整示例")
    print("=" * 60)
    
    # 1. 创建节点连接示例数据
    sample_data, lane_info = create_node_connection_example()
    
    # 2. 测试数据集
    success = test_node_connection_dataset()
    
    # 3. 可视化（可选）
    visualize_node_connections()
    
    if success:
        print("\n🎉 节点级连接规则示例完成！")
        print("\n数据格式说明:")
        print("node_connections 列格式: 'target1,type1;target2,type2'")
        print("连接类型:")
        print("  - direct: 直通连接")
        print("  - dashed: 虚线连接")
        print("  - solid: 实线连接（不连接）")
        print("\n使用方法:")
        print("python experiments/run_imputation.py \\")
        print("    --model-name spin \\")
        print("    --dataset-name lane_traffic \\")
        print("    --data-path node_connection_data.csv \\")
        print("    --config config/imputation/spin_lane.yaml")
    else:
        print("\n❌ 示例失败，请检查实现")


if __name__ == "__main__":
    main()

