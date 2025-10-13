"""
车道级交通数据预处理和格式转换工具
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta


class LaneDataProcessor:
    """车道级交通数据处理器"""
    
    def __init__(self, 
                 spatial_resolution: float = 10.0,  # 10米
                 temporal_resolution: int = 10,     # 10秒
                 speed_col: str = 'speed',
                 spacing_col: str = 'spacing',
                 time_col: str = 'timestamp',
                 lane_id_col: str = 'lane_id',
                 spatial_id_col: str = 'spatial_id',
                 lane_interaction_col: str = 'lane_interaction',  # 车道交互列
                 node_connections_col: str = 'node_connections'):  # 新增：节点连接列
        """
        初始化数据处理器
        
        Args:
            spatial_resolution: 空间分辨率（米）
            temporal_resolution: 时间分辨率（秒）
            speed_col: 速度列名
            spacing_col: 间距列名
            time_col: 时间列名
            lane_id_col: 车道ID列名
            spatial_id_col: 空间ID列名
        """
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        self.speed_col = speed_col
        self.spacing_col = spacing_col
        self.time_col = time_col
        self.lane_id_col = lane_id_col
        self.spatial_id_col = spatial_id_col
        self.lane_interaction_col = lane_interaction_col
        self.node_connections_col = node_connections_col
        
    def process_raw_data(self, 
                        raw_data: pd.DataFrame,
                        lane_info: Optional[Dict] = None) -> pd.DataFrame:
        """
        处理原始车道数据
        
        Args:
            raw_data: 原始数据DataFrame，包含车辆轨迹或检测数据
            lane_info: 车道信息字典，包含车道长度、位置等
            
        Returns:
            处理后的数据DataFrame
        """
        print("开始处理原始车道数据...")
        
        # 确保时间列为datetime类型
        raw_data[self.time_col] = pd.to_datetime(raw_data[self.time_col])
        
        # 按车道和时间分组
        processed_data = []
        
        for lane_id, lane_group in raw_data.groupby(self.lane_id_col):
            print(f"处理车道 {lane_id}，数据点: {len(lane_group)}")
            
            # 按时间排序
            lane_group = lane_group.sort_values(self.time_col)
            
            # 创建空间网格
            spatial_data = self._create_spatial_grid(lane_group, lane_id, lane_info)
            processed_data.append(spatial_data)
            
        # 合并所有车道数据
        result_df = pd.concat(processed_data, ignore_index=True)
        
        print(f"数据处理完成，总记录数: {len(result_df)}")
        return result_df
        
    def _create_spatial_grid(self, 
                            lane_data: pd.DataFrame, 
                            lane_id: str,
                            lane_info: Optional[Dict] = None) -> pd.DataFrame:
        """为单个车道创建空间网格"""
        
        # 获取车道长度（如果提供）
        lane_length = lane_info.get(lane_id, {}).get('length', None) if lane_info else None
        
        if lane_length is None:
            # 从数据中估算车道长度
            if 'position' in lane_data.columns:
                lane_length = lane_data['position'].max() - lane_data['position'].min()
            else:
                # 默认假设车道长度为1000米
                lane_length = 1000.0
                
        # 创建空间网格点
        n_spatial_points = int(lane_length / self.spatial_resolution)
        spatial_positions = np.linspace(0, lane_length, n_spatial_points)
        
        # 创建时间网格
        start_time = lane_data[self.time_col].min()
        end_time = lane_data[self.time_col].max()
        time_range = pd.date_range(start=start_time, end=end_time, 
                                 freq=f'{self.temporal_resolution}s')
        
        # 初始化结果数据
        grid_data = []
        
        for t, timestamp in enumerate(time_range):
            # 获取当前时间窗口的数据
            time_window_start = timestamp
            time_window_end = timestamp + timedelta(seconds=self.temporal_resolution)
            
            window_data = lane_data[
                (lane_data[self.time_col] >= time_window_start) & 
                (lane_data[self.time_col] < time_window_end)
            ]
            
            for s, spatial_pos in enumerate(spatial_positions):
                spatial_id = f"{lane_id}_{s:04d}"
                
                # 计算当前空间网格内的平均速度和间距
                spatial_window_start = spatial_pos - self.spatial_resolution / 2
                spatial_window_end = spatial_pos + self.spatial_resolution / 2
                
                if 'position' in window_data.columns:
                    spatial_mask = (
                        (window_data['position'] >= spatial_window_start) & 
                        (window_data['position'] < spatial_window_end)
                    )
                    spatial_window_data = window_data[spatial_mask]
                else:
                    # 如果没有位置信息，使用所有数据
                    spatial_window_data = window_data
                
                if len(spatial_window_data) > 0:
                    avg_speed = spatial_window_data[self.speed_col].mean()
                    avg_spacing = spatial_window_data[self.spacing_col].mean()
                else:
                    avg_speed = np.nan
                    avg_spacing = np.nan
                
                # 生成车道交互规则
                lane_interaction = self._generate_lane_interaction(
                    lane_id, spatial_pos, lane_info
                )
                
                # 生成节点连接规则
                node_connections = self._generate_node_connections(
                    spatial_id, lane_id, spatial_pos, lane_info
                )
                
                grid_data.append({
                    self.time_col: timestamp,
                    self.lane_id_col: lane_id,
                    self.spatial_id_col: spatial_id,
                    self.speed_col: avg_speed,
                    self.spacing_col: avg_spacing,
                    self.lane_interaction_col: lane_interaction,
                    self.node_connections_col: node_connections,
                    'spatial_position': spatial_pos,
                    'temporal_index': t,
                    'spatial_index': s
                })
                
        return pd.DataFrame(grid_data)
        
    def _generate_lane_interaction(self, lane_id: str, spatial_pos: float, 
                                  lane_info: Optional[Dict] = None) -> str:
        """
        生成车道交互规则
        
        Args:
            lane_id: 车道ID
            spatial_pos: 空间位置
            lane_info: 车道信息字典
            
        Returns:
            车道交互规则字符串
        """
        if lane_info and lane_id in lane_info:
            lane_data = lane_info[lane_id]
            
            # 检查是否有预定义的交互规则
            if 'interaction_rules' in lane_data:
                rules = lane_data['interaction_rules']
                for rule in rules:
                    start_pos, end_pos, interaction_type = rule
                    if start_pos <= spatial_pos <= end_pos:
                        return interaction_type
            
            # 检查是否有默认的交互类型
            if 'default_interaction' in lane_data:
                return lane_data['default_interaction']
        
        # 默认规则：基于车道ID和位置的简单规则
        # 这里可以根据实际需求实现更复杂的规则
        return self._get_default_interaction_rule(lane_id, spatial_pos)
        
    def _get_default_interaction_rule(self, lane_id: str, spatial_pos: float) -> str:
        """
        获取默认的车道交互规则
        
        这里实现一个简单的规则示例：
        - 相邻车道在特定位置可以交互（虚线）
        - 其他位置不允许交互（实线）
        """
        # 解析车道ID，假设格式为 "lane_X"
        try:
            lane_num = int(lane_id.split('_')[1])
        except:
            lane_num = 0
            
        # 简单的规则：每隔一定距离允许交互
        interaction_interval = 200.0  # 每200米
        interaction_zone = int(spatial_pos / interaction_interval) % 2 == 0
        
        if interaction_zone:
            return 'dashed'  # 虚线，可以交互
        else:
            return 'solid'   # 实线，不可以交互
            
    def create_lane_interaction_rules(self, 
                                    lane_ids: List[str],
                                    interaction_zones: List[Dict]) -> Dict:
        """
        创建车道交互规则
        
        Args:
            lane_ids: 车道ID列表
            interaction_zones: 交互区域列表，格式：
                [{'lanes': ['lane_0', 'lane_1'], 'start_pos': 0, 'end_pos': 500, 'type': 'dashed'}]
                
        Returns:
            车道交互规则字典
        """
        lane_rules = {}
        
        for lane_id in lane_ids:
            lane_rules[lane_id] = {
                'interaction_rules': [],
                'default_interaction': 'solid'  # 默认不允许交互
            }
            
        # 为每个交互区域添加规则
        for zone in interaction_zones:
            lanes = zone['lanes']
            start_pos = zone['start_pos']
            end_pos = zone['end_pos']
            interaction_type = zone['type']
            
            for lane_id in lanes:
                if lane_id in lane_rules:
                    lane_rules[lane_id]['interaction_rules'].append(
                        (start_pos, end_pos, interaction_type)
                    )
                    
        return lane_rules
        
    def _generate_node_connections(self, spatial_id: str, lane_id: str, 
                                 spatial_pos: float, lane_info: Optional[Dict] = None) -> str:
        """
        生成节点连接规则
        
        Args:
            spatial_id: 空间ID
            lane_id: 车道ID
            spatial_pos: 空间位置
            lane_info: 车道信息字典
            
        Returns:
            节点连接规则字符串
        """
        if lane_info and lane_id in lane_info:
            lane_data = lane_info[lane_id]
            
            # 检查是否有预定义的节点连接规则
            if 'node_connections' in lane_data:
                connections = lane_data['node_connections']
                if isinstance(connections, dict) and spatial_id in connections:
                    return self._format_node_connections(connections[spatial_id])
                elif isinstance(connections, list):
                    # 基于位置的规则匹配
                    for rule in connections:
                        if (rule['start_pos'] <= spatial_pos <= rule['end_pos'] and 
                            spatial_id in rule['connections']):
                            return self._format_node_connections(rule['connections'][spatial_id])
        
        # 默认规则：基于车道ID和位置的简单规则
        return self._get_default_node_connections(spatial_id, lane_id, spatial_pos)
        
    def _get_default_node_connections(self, spatial_id: str, lane_id: str, 
                                    spatial_pos: float) -> str:
        """
        获取默认的节点连接规则
        
        实现您提到的规则：
        - lane_0_0000 与 lane_0_0001 直通连接
        - lane_0_0000 与 lane_1_0000 虚线连接
        - lane_0_0000 与 lane_2_0032 虚线连接
        """
        connections = []
        
        # 解析当前节点信息
        try:
            current_lane_num = int(lane_id.split('_')[1])
            current_spatial_num = int(spatial_id.split('_')[2])
        except:
            return ""
        
        # 1. 同一车道内的相邻节点（直通连接）
        if current_spatial_num > 0:
            prev_spatial_id = f"{lane_id}_{current_spatial_num - 1:04d}"
            connections.append(f"{prev_spatial_id},direct")
        
        if current_spatial_num < 99:  # 假设最多100个空间节点
            next_spatial_id = f"{lane_id}_{current_spatial_num + 1:04d}"
            connections.append(f"{next_spatial_id},direct")
        
        # 2. 跨车道的连接（虚线连接）
        # 这里实现一个简单的规则：相邻车道在相同位置可以连接
        for other_lane_num in range(3):  # 假设有3个车道
            if other_lane_num != current_lane_num:
                other_lane_id = f"lane_{other_lane_num}"
                other_spatial_id = f"{other_lane_id}_{current_spatial_num:04d}"
                connections.append(f"{other_spatial_id},dashed")
        
        return ";".join(connections)
        
    def _format_node_connections(self, connections):
        """格式化节点连接规则"""
        if isinstance(connections, dict):
            return ";".join([f"{target},{conn_type}" for target, conn_type in connections.items()])
        elif isinstance(connections, str):
            return connections
        else:
            return ""
            
    def create_node_connection_rules(self, 
                                   spatial_ids: List[str],
                                   connection_rules: List[Dict]) -> Dict:
        """
        创建节点连接规则
        
        Args:
            spatial_ids: 空间ID列表
            connection_rules: 连接规则列表，格式：
                [{'spatial_id': 'lane_0_0000', 'connections': {'lane_0_0001': 'direct', 'lane_1_0000': 'dashed'}}]
                
        Returns:
            节点连接规则字典
        """
        node_rules = {}
        
        for spatial_id in spatial_ids:
            node_rules[spatial_id] = {
                'connections': {},
                'default_connections': []
            }
            
        # 为每个连接规则添加连接
        for rule in connection_rules:
            spatial_id = rule['spatial_id']
            connections = rule['connections']
            
            if spatial_id in node_rules:
                node_rules[spatial_id]['connections'] = connections
                
        return node_rules
        
    def create_sample_data(self, 
                          n_lanes: int = 5,
                          lane_length: float = 1000.0,
                          time_hours: float = 24.0,
                          seed: int = 42) -> pd.DataFrame:
        """
        创建示例车道数据用于测试
        
        Args:
            n_lanes: 车道数量
            lane_length: 车道长度（米）
            time_hours: 时间长度（小时）
            seed: 随机种子
            
        Returns:
            示例数据DataFrame
        """
        np.random.seed(seed)
        
        # 创建时间序列
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        time_range = pd.date_range(
            start=start_time, 
            periods=int(time_hours * 3600 / self.temporal_resolution),
            freq=f'{self.temporal_resolution}s'
        )
        
        # 创建空间网格
        n_spatial_points = int(lane_length / self.spatial_resolution)
        spatial_positions = np.linspace(0, lane_length, n_spatial_points)
        
        data = []
        
        for lane_id in range(n_lanes):
            print(f"生成车道 {lane_id} 的示例数据...")
            
            for t, timestamp in enumerate(time_range):
                for s, spatial_pos in enumerate(spatial_positions):
                    spatial_id = f"lane_{lane_id}_{s:04d}"
                    
                    # 生成模拟的速度和间距数据
                    # 添加时间周期性（早晚高峰）
                    hour = timestamp.hour
                    time_factor = 1.0 + 0.5 * np.sin(2 * np.pi * hour / 24)
                    
                    # 添加空间变化（车道内不同位置）
                    spatial_factor = 1.0 + 0.3 * np.sin(2 * np.pi * spatial_pos / lane_length)
                    
                    # 添加随机噪声
                    noise = np.random.normal(0, 0.1)
                    
                    # 生成速度和间距（考虑相关性）
                    base_speed = 30.0 * time_factor * spatial_factor + noise * 5
                    base_spacing = 20.0 / time_factor + noise * 2
                    
                    # 确保数据在合理范围内
                    speed = np.clip(base_speed, 0, 60)
                    spacing = np.clip(base_spacing, 5, 50)
                    
                    # 生成车道交互规则
                    lane_interaction = self._get_default_interaction_rule(f"lane_{lane_id}", spatial_pos)
                    
                    # 生成节点连接规则
                    node_connections = self._get_default_node_connections(
                        spatial_id, f"lane_{lane_id}", spatial_pos
                    )
                    
                    data.append({
                        self.time_col: timestamp,
                        self.lane_id_col: f"lane_{lane_id}",
                        self.spatial_id_col: spatial_id,
                        self.speed_col: speed,
                        self.spacing_col: spacing,
                        self.lane_interaction_col: lane_interaction,
                        self.node_connections_col: node_connections,
                        'spatial_position': spatial_pos,
                        'temporal_index': t,
                        'spatial_index': s
                    })
                    
        return pd.DataFrame(data)
        
    def save_data(self, 
                  data: pd.DataFrame, 
                  save_path: str,
                  format: str = 'csv'):
        """
        保存处理后的数据
        
        Args:
            data: 数据DataFrame
            save_path: 保存路径
            format: 保存格式 ('csv', 'npz', 'pkl')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            data.to_csv(save_path, index=False)
        elif format == 'npz':
            np.savez(save_path,
                    timestamps=data[self.time_col].values,
                    lane_ids=data[self.lane_id_col].values,
                    spatial_ids=data[self.spatial_id_col].values,
                    speeds=data[self.speed_col].values,
                    spacings=data[self.spacing_col].values)
        elif format == 'pkl':
            data.to_pickle(save_path)
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        print(f"数据已保存到: {save_path}")
        
    def create_lane_info(self, 
                        lane_ids: List[str],
                        lane_lengths: Optional[List[float]] = None,
                        lane_positions: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        创建车道信息字典
        
        Args:
            lane_ids: 车道ID列表
            lane_lengths: 车道长度列表
            lane_positions: 车道位置列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            车道信息字典
        """
        lane_info = {}
        
        for i, lane_id in enumerate(lane_ids):
            lane_info[lane_id] = {
                'length': lane_lengths[i] if lane_lengths else 1000.0,
                'position': lane_positions[i] if lane_positions else (0.0, 0.0),
                'spatial_resolution': self.spatial_resolution,
                'temporal_resolution': self.temporal_resolution
            }
            
        return lane_info
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据格式和完整性
        
        Args:
            data: 数据DataFrame
            
        Returns:
            验证是否通过
        """
        required_cols = [self.time_col, self.lane_id_col, self.spatial_id_col, 
                        self.speed_col, self.spacing_col]
        
        # 检查必需列
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"缺少必需列: {missing_cols}")
            return False
            
        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(data[self.time_col]):
            print(f"时间列 {self.time_col} 不是datetime类型")
            return False
            
        # 检查数值范围
        if data[self.speed_col].min() < 0 or data[self.speed_col].max() > 200:
            print(f"速度值超出合理范围: [{data[self.speed_col].min()}, {data[self.speed_col].max()}]")
            
        if data[self.spacing_col].min() < 0 or data[self.spacing_col].max() > 100:
            print(f"间距值超出合理范围: [{data[self.spacing_col].min()}, {data[self.spacing_col].max()}]")
            
        # 检查缺失值
        missing_ratio = data[required_cols].isnull().sum().sum() / (len(data) * len(required_cols))
        print(f"缺失值比例: {missing_ratio:.3f}")
        
        return True


def create_sample_dataset(output_path: str = "sample_lane_data.csv"):
    """创建示例数据集"""
    processor = LaneDataProcessor()
    
    # 创建示例数据
    sample_data = processor.create_sample_data(
        n_lanes=3,
        lane_length=1000.0,
        time_hours=2.0,  # 2小时数据用于测试
        seed=42
    )
    
    # 保存数据
    processor.save_data(sample_data, output_path, format='csv')
    
    # 验证数据
    if processor.validate_data(sample_data):
        print("✅ 示例数据集创建成功")
    else:
        print("❌ 示例数据集验证失败")
        
    return sample_data


if __name__ == "__main__":
    # 创建示例数据集
    sample_data = create_sample_dataset("sample_lane_data.csv")
    print(f"示例数据形状: {sample_data.shape}")
    print(f"列名: {sample_data.columns.tolist()}")
    print(f"时间范围: {sample_data['timestamp'].min()} 到 {sample_data['timestamp'].max()}")
    print(f"车道数: {sample_data['lane_id'].nunique()}")
    print(f"空间节点数: {sample_data['spatial_id'].nunique()}")
