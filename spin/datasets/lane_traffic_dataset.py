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
                 static_data_path: str,
                 dynamic_data_path: str,
                 speed_col: str = 'speed',
                 spacing_col: str = 'spacing',
                 time_col: str = 'timestamp',
                 lane_id_col: str = 'lane_id',
                 spatial_id_col: str = 'spatial_id',
                 node_connections_col: str = 'node_connections',
                 mask_data_path: Optional[str] = None,
                 window_size: int = 12,
                 stride: int = 1,
                 val_len: float = 0.1,
                 test_len: float = 0.2,
                 impute_nans: bool = True,
                 **kwargs):
        """
        初始化车道级交通数据集
        
        数据格式：采用静态道路数据和动态交通数据分离的设计
        - 静态道路数据：包含 lane_id, spatial_id, node_connections
        - 动态交通数据：包含 timestamp, spatial_id, speed, spacing
        
        Args:
            static_data_path: 静态道路数据文件路径（CSV, JSON, PKL）
            dynamic_data_path: 动态交通数据文件路径（CSV, NPZ, PKL）
            speed_col: 速度列名
            spacing_col: 间距列名  
            time_col: 时间列名
            lane_id_col: 车道ID列名
            spatial_id_col: 空间ID列名
            node_connections_col: 节点连接列名
            mask_data_path: 用户自定义掩码文件路径（可选），支持CSV, NPZ, PKL格式
                - CSV格式：需包含 timestamp, spatial_id, is_observed 列
                - NPZ格式：需包含 'mask' 数组，形状为 [n_times, n_spaces] 或 [n_times, n_spaces, n_features]
                - PKL格式：包含掩码矩阵或字典
                如果不提供，则使用随机生成的掩码
            window_size: 时间窗口大小
            stride: 时间步长
            val_len: 验证集比例
            test_len: 测试集比例
            impute_nans: 是否填充缺失值
        """
        super().__init__(**kwargs)
        
        self.static_data_path = static_data_path
        self.dynamic_data_path = dynamic_data_path
        self.speed_col = speed_col
        self.spacing_col = spacing_col
        self.time_col = time_col
        self.lane_id_col = lane_id_col
        self.spatial_id_col = spatial_id_col
        self.node_connections_col = node_connections_col
        self.mask_data_path = mask_data_path
        self.window_size = window_size
        self.stride = stride
        self.val_len = val_len
        self.test_len = test_len
        self.impute_nans = impute_nans
        
        # 加载和预处理数据
        self._load_data()
        self._preprocess_data()
        
    def _load_data(self):
        """加载静态道路数据和动态交通数据"""
        # 1. 加载静态道路数据
        static_path = Path(self.static_data_path)
        if static_path.suffix == '.csv':
            self.static_df = pd.read_csv(static_path)
        elif static_path.suffix == '.json':
            with open(static_path, 'r') as f:
                static_data = json.load(f)
            if 'nodes' in static_data:
                self.static_df = pd.DataFrame(static_data['nodes'])
            else:
                self.static_df = pd.DataFrame(static_data)
        elif static_path.suffix == '.pkl':
            self.static_df = pd.read_pickle(static_path)
        else:
            raise ValueError(f"不支持的静态数据文件格式: {static_path.suffix}")
            
        print(f"✅ 加载静态道路数据: {self.static_df.shape[0]} 个节点")
        
        # 2. 加载动态交通数据
        dynamic_path = Path(self.dynamic_data_path)
        if dynamic_path.suffix == '.csv':
            self.dynamic_df = pd.read_csv(dynamic_path)
        elif dynamic_path.suffix == '.npz':
            data = np.load(dynamic_path)
            self.dynamic_df = pd.DataFrame({
                self.time_col: data['timestamps'],
                self.spatial_id_col: data['spatial_ids'],
                self.speed_col: data['speeds'],
                self.spacing_col: data['spacings']
            })
        elif dynamic_path.suffix == '.pkl':
            self.dynamic_df = pd.read_pickle(dynamic_path)
        else:
            raise ValueError(f"不支持的动态数据文件格式: {dynamic_path.suffix}")
            
        print(f"✅ 加载动态交通数据: {self.dynamic_df.shape[0]} 条记录")
        
        # 3. 验证数据一致性
        static_spatial_ids = set(self.static_df[self.spatial_id_col])
        dynamic_spatial_ids = set(self.dynamic_df[self.spatial_id_col])
        
        if not dynamic_spatial_ids.issubset(static_spatial_ids):
            missing = dynamic_spatial_ids - static_spatial_ids
            raise ValueError(f"动态数据中有 {len(missing)} 个 spatial_id 在静态数据中不存在")
        
        print(f"✅ 数据一致性验证通过")
        
    def _preprocess_data(self):
        """数据预处理"""
        # 确保动态数据的时间列为datetime类型
        self.dynamic_df[self.time_col] = pd.to_datetime(self.dynamic_df[self.time_col])
        
        # 按时间和空间ID排序
        self.dynamic_df = self.dynamic_df.sort_values([self.time_col, self.spatial_id_col])
        
        # 从动态数据创建唯一的时间戳索引
        self.timestamps = self.dynamic_df[self.time_col].unique()
        self.timestamps = pd.to_datetime(self.timestamps).sort_values()
        
        # 从静态数据创建唯一的空间ID索引
        self.spatial_ids = self.static_df[self.spatial_id_col].values
        self.spatial_ids = np.sort(self.spatial_ids)
        
        # 从静态数据创建唯一的车道ID索引
        self.lane_ids = self.static_df[self.lane_id_col].unique()
        self.lane_ids = np.sort(self.lane_ids)
        
        print(f"时间步数: {len(self.timestamps)}")
        print(f"空间节点数: {len(self.spatial_ids)}")
        print(f"车道数: {len(self.lane_ids)}")
        
        # 构建时空数据矩阵（从动态数据）
        self._build_spatiotemporal_matrix()
        
        # 构建图连接（从静态数据）
        self._build_graph_connectivity()
        
        # 创建训练/验证/测试掩码
        self._create_masks()
        
    def _build_spatiotemporal_matrix(self):
        """构建时空数据矩阵（从动态交通数据）"""
        # 创建时空数据矩阵 [时间, 空间, 特征]
        n_times = len(self.timestamps)
        n_spaces = len(self.spatial_ids)
        n_features = 2  # 速度和间距
        
        self.data = np.full((n_times, n_spaces, n_features), np.nan)
        
        # 创建spatial_id到索引的映射
        spatial_id_to_idx = {sid: idx for idx, sid in enumerate(self.spatial_ids)}
        
        # 填充数据（从动态数据）
        for _, row in self.dynamic_df.iterrows():
            time_idx = np.where(self.timestamps == row[self.time_col])[0][0]
            space_idx = spatial_id_to_idx.get(row[self.spatial_id_col])
            
            if space_idx is not None:
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
        """构建基于节点级连接规则的图连接（从静态道路数据）"""
        # 从静态数据创建空间ID到车道ID和连接规则的映射
        spatial_to_lane = {}
        spatial_to_connections = {}
        
        for _, row in self.static_df.iterrows():
            spatial_to_lane[row[self.spatial_id_col]] = row[self.lane_id_col]
            
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
        
        self.adjacency = adj_matrix
        print(f"图连接矩阵形状: {self.adjacency.shape}")
        print(f"连接数: {np.sum(adj_matrix > 0) // 2}")  # 除以2因为是无向图
        
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
                    if connection_type in ['direct', 'straight']:
                        weight = 1.0
                    elif connection_type == 'dashed':
                        weight = 0.5
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
        
        # 如果提供了用户自定义掩码，则从文件加载
        if self.mask_data_path is not None:
            self._load_user_mask()
            print(f"✅ 使用用户自定义掩码")
            print(f"   已观测数据比例: {self.training_mask.mean():.3f}")
            print(f"   未观测数据比例: {self.eval_mask.mean():.3f}")
        else:
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
            print(f"✅ 使用随机生成的掩码")
            
    def _load_user_mask(self):
        """从用户提供的文件加载掩码数据"""
        n_times, n_spaces, n_features = self.data.shape
        mask_path = Path(self.mask_data_path)
        
        if not mask_path.exists():
            raise ValueError(f"掩码文件不存在: {self.mask_data_path}")
        
        # 根据文件格式加载掩码
        if mask_path.suffix == '.csv':
            # CSV格式：包含 timestamp, spatial_id, is_observed 列
            mask_df = pd.read_csv(mask_path)
            required_cols = [self.time_col, self.spatial_id_col, 'is_observed']
            
            if not all(col in mask_df.columns for col in required_cols):
                raise ValueError(f"CSV掩码文件必须包含以下列: {required_cols}")
            
            # 初始化掩码矩阵（默认所有位置都是未观测的）
            self.training_mask = np.zeros((n_times, n_spaces, n_features), dtype=bool)
            
            # 创建spatial_id到索引的映射
            spatial_id_to_idx = {sid: idx for idx, sid in enumerate(self.spatial_ids)}
            
            # 将时间戳转换为datetime
            mask_df[self.time_col] = pd.to_datetime(mask_df[self.time_col])
            
            # 填充掩码
            for _, row in mask_df.iterrows():
                timestamp = row[self.time_col]
                spatial_id = row[self.spatial_id_col]
                is_observed = bool(row['is_observed'])
                
                # 查找时间索引
                time_matches = np.where(self.timestamps == timestamp)[0]
                if len(time_matches) == 0:
                    continue
                time_idx = time_matches[0]
                
                # 查找空间索引
                space_idx = spatial_id_to_idx.get(spatial_id)
                if space_idx is None:
                    continue
                
                # 设置掩码（对所有特征都使用相同的掩码）
                self.training_mask[time_idx, space_idx, :] = is_observed
                
        elif mask_path.suffix == '.npz':
            # NPZ格式：包含 'mask' 数组
            data = np.load(mask_path)
            
            if 'mask' not in data:
                raise ValueError("NPZ掩码文件必须包含 'mask' 数组")
            
            mask = data['mask']
            
            # 检查形状是否匹配
            if mask.shape == (n_times, n_spaces):
                # 二维掩码：扩展到所有特征
                self.training_mask = np.repeat(mask[:, :, np.newaxis], n_features, axis=2)
            elif mask.shape == (n_times, n_spaces, n_features):
                # 三维掩码：直接使用
                self.training_mask = mask.astype(bool)
            else:
                raise ValueError(f"NPZ掩码形状不匹配。期望 ({n_times}, {n_spaces}) 或 ({n_times}, {n_spaces}, {n_features})，实际 {mask.shape}")
                
        elif mask_path.suffix == '.pkl':
            # PKL格式：包含掩码矩阵或字典
            with open(mask_path, 'rb') as f:
                mask_data = pickle.load(f)
            
            if isinstance(mask_data, np.ndarray):
                mask = mask_data
                
                # 检查形状是否匹配
                if mask.shape == (n_times, n_spaces):
                    # 二维掩码：扩展到所有特征
                    self.training_mask = np.repeat(mask[:, :, np.newaxis], n_features, axis=2)
                elif mask.shape == (n_times, n_spaces, n_features):
                    # 三维掩码：直接使用
                    self.training_mask = mask.astype(bool)
                else:
                    raise ValueError(f"PKL掩码形状不匹配。期望 ({n_times}, {n_spaces}) 或 ({n_times}, {n_spaces}, {n_features})，实际 {mask.shape}")
                    
            elif isinstance(mask_data, dict):
                # 字典格式：包含 'training_mask' 和/或 'eval_mask'
                if 'training_mask' in mask_data:
                    self.training_mask = mask_data['training_mask'].astype(bool)
                else:
                    raise ValueError("PKL字典必须包含 'training_mask' 键")
            else:
                raise ValueError(f"PKL掩码文件格式不支持: {type(mask_data)}")
        else:
            raise ValueError(f"不支持的掩码文件格式: {mask_path.suffix}")
        
        # 创建评估掩码（与训练掩码相反）
        self.eval_mask = ~self.training_mask
        
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
