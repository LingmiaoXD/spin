"""
车道级交通状况数据集类
支持10m×10s时空网格的平均速度和间距数据，节点间通过车道关联进行图连接
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple, Union, List, Dict, Any
from pathlib import Path
import pickle
import json
from tsl.datasets import Dataset
from tsl.data import SpatioTemporalDataset
from tsl.data.preprocessing import StandardScaler
from tsl.ops.connectivity import adj_to_edge_index
from tsl.utils.python_utils import ensure_list


class LaneTrafficDataset(Dataset):
    """
    车道级交通状况数据集
    
    数据格式：
    - 每个节点代表10m×10s的时空网格
    - 节点属性：平均速度、平均间距
    - 图连接：基于车道关联的空间关系
    """
    
    def __init__(self, 
                 data_path: str,
                 speed_col: str = 'speed',
                 spacing_col: str = 'spacing',
                 time_col: str = 'timestamp',
                 lane_id_col: str = 'lane_id',
                 spatial_id_col: str = 'spatial_id',
                 lane_interaction_col: str = 'lane_interaction',  # 车道交互列
                 node_connections_col: str = 'node_connections',  # 新增：节点连接列
                 window_size: int = 12,
                 stride: int = 1,
                 val_len: float = 0.1,
                 test_len: float = 0.2,
                 impute_nans: bool = True,
                 **kwargs):
        """
        初始化车道级交通数据集
        
        Args:
            data_path: 数据文件路径 (支持CSV, NPZ, PKL格式)
            speed_col: 速度列名
            spacing_col: 间距列名  
            time_col: 时间列名
            lane_id_col: 车道ID列名
            spatial_id_col: 空间ID列名
            window_size: 时间窗口大小
            stride: 时间步长
            val_len: 验证集比例
            test_len: 测试集比例
            impute_nans: 是否填充缺失值
        """
        super().__init__(**kwargs)
        
        self.data_path = data_path
        self.speed_col = speed_col
        self.spacing_col = spacing_col
        self.time_col = time_col
        self.lane_id_col = lane_id_col
        self.spatial_id_col = spatial_id_col
        self.lane_interaction_col = lane_interaction_col
        self.node_connections_col = node_connections_col
        self.window_size = window_size
        self.stride = stride
        self.val_len = val_len
        self.test_len = test_len
        self.impute_nans = impute_nans
        
        # 加载和预处理数据
        self._load_data()
        self._preprocess_data()
        
    def _load_data(self):
        """加载原始数据"""
        data_path = Path(self.data_path)
        
        if data_path.suffix == '.csv':
            self.df = pd.read_csv(data_path)
        elif data_path.suffix == '.npz':
            data = np.load(data_path)
            self.df = pd.DataFrame({
                self.time_col: data['timestamps'],
                self.lane_id_col: data['lane_ids'],
                self.spatial_id_col: data['spatial_ids'],
                self.speed_col: data['speeds'],
                self.spacing_col: data['spacings']
            })
        elif data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.df = pd.DataFrame(data)
        else:
            raise ValueError(f"不支持的文件格式: {data_path.suffix}")
            
        print(f"加载数据完成: {self.df.shape[0]} 条记录")
        
    def _preprocess_data(self):
        """数据预处理"""
        # 确保时间列为datetime类型
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        
        # 按时间和空间ID排序
        self.df = self.df.sort_values([self.time_col, self.spatial_id_col])
        
        # 创建唯一的时间戳索引
        self.timestamps = self.df[self.time_col].unique()
        self.timestamps = pd.to_datetime(self.timestamps).sort_values()
        
        # 创建唯一的空间ID索引
        self.spatial_ids = self.df[self.spatial_id_col].unique()
        self.spatial_ids = np.sort(self.spatial_ids)
        
        # 创建唯一的车道ID索引
        self.lane_ids = self.df[self.lane_id_col].unique()
        self.lane_ids = np.sort(self.lane_ids)
        
        print(f"时间步数: {len(self.timestamps)}")
        print(f"空间节点数: {len(self.spatial_ids)}")
        print(f"车道数: {len(self.lane_ids)}")
        
        # 构建时空数据矩阵
        self._build_spatiotemporal_matrix()
        
        # 构建图连接
        self._build_graph_connectivity()
        
        # 创建训练/验证/测试掩码
        self._create_masks()
        
    def _build_spatiotemporal_matrix(self):
        """构建时空数据矩阵"""
        # 创建时空数据矩阵 [时间, 空间, 特征]
        n_times = len(self.timestamps)
        n_spaces = len(self.spatial_ids)
        n_features = 2  # 速度和间距
        
        self.data = np.full((n_times, n_spaces, n_features), np.nan)
        
        # 填充数据
        for _, row in self.df.iterrows():
            time_idx = np.where(self.timestamps == row[self.time_col])[0][0]
            space_idx = np.where(self.spatial_ids == row[self.spatial_id_col])[0][0]
            
            self.data[time_idx, space_idx, 0] = row[self.speed_col]  # 速度
            self.data[time_idx, space_idx, 1] = row[self.spacing_col]  # 间距
            
        # 处理缺失值
        if self.impute_nans:
            # 使用前向填充
            for i in range(1, n_times):
                mask = np.isnan(self.data[i])
                self.data[i][mask] = self.data[i-1][mask]
                
        print(f"数据矩阵形状: {self.data.shape}")
        print(f"缺失值比例: {np.isnan(self.data).mean():.3f}")
        
    def _build_graph_connectivity(self):
        """构建基于节点级连接规则的图连接"""
        # 创建空间ID到车道ID的映射
        spatial_to_lane = {}
        spatial_to_interaction = {}
        spatial_to_connections = {}  # 新增：空间ID到节点连接的映射
        
        for _, row in self.df.iterrows():
            spatial_to_lane[row[self.spatial_id_col]] = row[self.lane_id_col]
            
            # 获取车道交互信息
            if self.lane_interaction_col in row and pd.notna(row[self.lane_interaction_col]):
                spatial_to_interaction[row[self.spatial_id_col]] = row[self.lane_interaction_col]
            
            # 获取节点连接信息
            if self.node_connections_col in row and pd.notna(row[self.node_connections_col]):
                spatial_to_connections[row[self.spatial_id_col]] = row[self.node_connections_col]
            
        # 构建邻接矩阵
        n_spaces = len(self.spatial_ids)
        adj_matrix = np.zeros((n_spaces, n_spaces))
        
        # 1. 同一车道内的节点连接（纵向连接）
        for lane_id in self.lane_ids:
            lane_spaces = [spatial_id for spatial_id, l_id in spatial_to_lane.items() 
                          if l_id == lane_id]
            lane_spaces = sorted(lane_spaces)
            
            # 按空间顺序连接相邻节点
            for i in range(len(lane_spaces) - 1):
                idx1 = np.where(self.spatial_ids == lane_spaces[i])[0][0]
                idx2 = np.where(self.spatial_ids == lane_spaces[i+1])[0][0]
                adj_matrix[idx1, idx2] = 1.0
                adj_matrix[idx2, idx1] = 1.0  # 无向图
                
        # 2. 基于节点连接规则的连接
        self._add_node_connections(adj_matrix, spatial_to_connections)
        
        # 3. 基于车道交互规则的连接（向后兼容）
        if not spatial_to_connections:  # 如果没有节点连接规则，使用车道交互规则
            self._add_cross_lane_connections(adj_matrix, spatial_to_lane, spatial_to_interaction)
        
        self.adjacency = adj_matrix
        print(f"图连接矩阵形状: {self.adjacency.shape}")
        print(f"连接数: {np.sum(adj_matrix > 0)}")
        
    def _add_cross_lane_connections(self, adj_matrix, spatial_to_lane, spatial_to_interaction):
        """添加跨车道连接，基于车道间交互规则"""
        n_spaces = len(self.spatial_ids)
        
        # 获取所有车道对
        lane_pairs = []
        for i, lane1 in enumerate(self.lane_ids):
            for j, lane2 in enumerate(self.lane_ids):
                if i < j:  # 避免重复
                    lane_pairs.append((lane1, lane2))
        
        # 为每个车道对添加连接
        for lane1, lane2 in lane_pairs:
            lane1_spaces = [spatial_id for spatial_id, l_id in spatial_to_lane.items() 
                           if l_id == lane1]
            lane2_spaces = [spatial_id for spatial_id, l_id in spatial_to_lane.items() 
                           if l_id == lane2]
            
            # 按空间位置排序
            lane1_spaces = sorted(lane1_spaces)
            lane2_spaces = sorted(lane2_spaces)
            
            # 检查每个空间位置的交互规则
            for space1 in lane1_spaces:
                for space2 in lane2_spaces:
                    # 检查是否允许交互
                    if self._can_lanes_interact(space1, space2, spatial_to_interaction):
                        idx1 = np.where(self.spatial_ids == space1)[0][0]
                        idx2 = np.where(self.spatial_ids == space2)[0][0]
                        
                        # 添加双向连接
                        adj_matrix[idx1, idx2] = 1.0
                        adj_matrix[idx2, idx1] = 1.0
                        
    def _can_lanes_interact(self, space1, space2, spatial_to_interaction):
        """判断两个空间位置的车道是否可以交互"""
        # 如果两个空间位置都有交互规则信息
        if space1 in spatial_to_interaction and space2 in spatial_to_interaction:
            interaction1 = spatial_to_interaction[space1]
            interaction2 = spatial_to_interaction[space2]
            
            # 检查交互规则
            # 支持多种格式：
            # 1. 字符串格式：'dashed'/'solid' 或 'interact'/'no_interact'
            # 2. 数值格式：1表示可交互，0表示不可交互
            # 3. 字典格式：{'interaction': True/False, 'type': 'dashed'/'solid'}
            
            if isinstance(interaction1, str):
                can_interact1 = interaction1.lower() in ['dashed', 'interact', 'true', '1']
            elif isinstance(interaction1, (int, float)):
                can_interact1 = bool(interaction1)
            elif isinstance(interaction1, dict):
                can_interact1 = interaction1.get('interaction', False)
            else:
                can_interact1 = False
                
            if isinstance(interaction2, str):
                can_interact2 = interaction2.lower() in ['dashed', 'interact', 'true', '1']
            elif isinstance(interaction2, (int, float)):
                can_interact2 = bool(interaction2)
            elif isinstance(interaction2, dict):
                can_interact2 = interaction2.get('interaction', False)
            else:
                can_interact2 = False
                
            # 两个位置都允许交互才能建立连接
            return can_interact1 and can_interact2
        
        # 如果没有交互规则信息，默认不允许交互
        return False
        
    def _add_node_connections(self, adj_matrix, spatial_to_connections):
        """添加基于节点连接规则的连接"""
        n_spaces = len(self.spatial_ids)
        
        # 创建空间ID到索引的映射
        spatial_to_idx = {spatial_id: idx for idx, spatial_id in enumerate(self.spatial_ids)}
        
        # 处理每个节点的连接规则
        for spatial_id, connections in spatial_to_connections.items():
            if spatial_id not in spatial_to_idx:
                continue
                
            source_idx = spatial_to_idx[spatial_id]
            
            # 解析连接规则
            connections = self._parse_node_connections(connections)
            
            for target_spatial_id, connection_type in connections.items():
                if target_spatial_id in spatial_to_idx:
                    target_idx = spatial_to_idx[target_spatial_id]
                    
                    # 根据连接类型设置权重
                    if connection_type in ['direct', 'straight', 'dashed']:
                        weight = 1.0
                    elif connection_type in ['indirect', 'solid']:
                        weight = 0.0  # 不连接
                    else:
                        weight = 0.0
                    
                    # 添加双向连接
                    if weight > 0:
                        adj_matrix[source_idx, target_idx] = weight
                        adj_matrix[target_idx, source_idx] = weight
                        
    def _parse_node_connections(self, connections):
        """
        解析节点连接规则
        
        支持多种格式：
        1. 字符串格式：'lane_0_0001,dashed;lane_1_0000,dashed;lane_2_0032,dashed'
        2. 字典格式：{'lane_0_0001': 'dashed', 'lane_1_0000': 'dashed'}
        3. JSON格式：'{"lane_0_0001": "dashed", "lane_1_0000": "dashed"}'
        """
        import json
        
        if isinstance(connections, dict):
            return connections
        elif isinstance(connections, str):
            # 尝试解析JSON格式
            try:
                return json.loads(connections)
            except:
                # 解析分号分隔的格式
                result = {}
                for connection in connections.split(';'):
                    if ',' in connection:
                        target, conn_type = connection.strip().split(',', 1)
                        result[target.strip()] = conn_type.strip()
                return result
        else:
            return {}
        
    def _create_masks(self):
        """创建训练/验证/测试掩码"""
        n_times, n_spaces, n_features = self.data.shape
        
        # 创建训练掩码（所有数据用于训练）
        self.training_mask = np.ones((n_times, n_spaces, n_features), dtype=bool)
        
        # 创建评估掩码（随机选择20%的数据点用于评估）
        np.random.seed(42)
        eval_indices = np.random.choice(
            n_times * n_spaces * n_features,
            size=int(0.2 * n_times * n_spaces * n_features),
            replace=False
        )
        
        self.eval_mask = np.zeros((n_times, n_spaces, n_features), dtype=bool)
        eval_mask_flat = self.eval_mask.reshape(-1)
        eval_mask_flat[eval_indices] = True
        
    def get_connectivity(self, threshold: float = 0.1, 
                        include_self: bool = False,
                        force_symmetric: bool = True) -> np.ndarray:
        """获取图连接矩阵"""
        adj = self.adjacency.copy()
        
        if not include_self:
            np.fill_diagonal(adj, 0)
            
        if force_symmetric:
            adj = (adj + adj.T) / 2
            
        return adj
        
    def numpy(self, return_idx: bool = False) -> Union[Tuple, np.ndarray]:
        """返回numpy格式的数据"""
        if return_idx:
            return self.data, self.timestamps, self.spatial_ids
        return self.data
        
    def datetime_encoded(self, encoding: List[str]) -> pd.DataFrame:
        """获取时间编码"""
        df = pd.DataFrame({'datetime': self.timestamps})
        
        if 'day' in encoding:
            df['day'] = df['datetime'].dt.dayofweek
        if 'week' in encoding:
            df['week'] = df['datetime'].dt.isocalendar().week
        if 'hour' in encoding:
            df['hour'] = df['datetime'].dt.hour
        if 'minute' in encoding:
            df['minute'] = df['datetime'].dt.minute
            
        return df
        
    def get_splitter(self, val_len: float = None, test_len: float = None):
        """获取数据分割器"""
        from tsl.data.splitter import TemporalSplitter
        
        val_len = val_len or self.val_len
        test_len = test_len or self.test_len
        
        return TemporalSplitter(val_len=val_len, test_len=test_len)
        
    @property
    def n_nodes(self) -> int:
        """节点数量"""
        return len(self.spatial_ids)
        
    @property
    def n_channels(self) -> int:
        """特征通道数"""
        return self.data.shape[-1]
        
    @property
    def length(self) -> int:
        """时间序列长度"""
        return len(self.timestamps)
        
    def __len__(self) -> int:
        return self.length
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个时间步的数据"""
        return {
            'data': self.data[idx],
            'timestamp': self.timestamps[idx],
            'spatial_ids': self.spatial_ids
        }
        
    def save_processed_data(self, save_path: str):
        """保存预处理后的数据"""
        save_data = {
            'data': self.data,
            'timestamps': self.timestamps,
            'spatial_ids': self.spatial_ids,
            'lane_ids': self.lane_ids,
            'adjacency': self.adjacency,
            'training_mask': self.training_mask,
            'eval_mask': self.eval_mask
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"预处理数据已保存到: {save_path}")
        
    @classmethod
    def load_processed_data(cls, load_path: str, **kwargs):
        """加载预处理后的数据"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            
        # 创建临时数据集实例
        dataset = cls.__new__(cls)
        dataset.data = data['data']
        dataset.timestamps = data['timestamps']
        dataset.spatial_ids = data['spatial_ids']
        dataset.lane_ids = data['lane_ids']
        dataset.adjacency = data['adjacency']
        dataset.training_mask = data['training_mask']
        dataset.eval_mask = data['eval_mask']
        
        return dataset
