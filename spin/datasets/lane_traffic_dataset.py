"""
车道级交通状况数据集类
支持静态道路数据(graph.json)和动态交通数据(csv)，以及用户自定义掩码
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
    - 静态数据(graph.json): 包含 lane_id 和 node_connections
    - 动态数据(csv): 包含 lane_id, start_frame, avg_speed, avg_occupancy 等特征
    - 掩码数据(csv): 包含 start_frame, lane_id, is_observed
    """
    
    # 默认特征列（可配置）
    DEFAULT_FEATURE_COLS = [
        'avg_speed', 'avg_occupancy', 'total_vehicles', 
        'car_ratio', 'medium_ratio', 'heavy_ratio', 'motorcycle_ratio',
        'crossing_ratio', 'direct_ratio', 'near_ratio'
    ]
    
    def __init__(self, 
                 static_data_path: str,
                 dynamic_data_path: str,
                 mask_data_path: Optional[str] = None,
                 feature_cols: Optional[List[str]] = None,
                 time_col: str = 'start_frame',
                 lane_id_col: str = 'lane_id',
                 mask_time_col: str = 'start_frame',
                 mask_lane_col: str = 'lane_id',
                 mask_value_col: str = 'is_observed',
                 window_size: int = 12,
                 stride: int = 1,
                 val_len: float = 0.1,
                 test_len: float = 0.2,
                 impute_nans: bool = True,
                 fill_value: float = 0.0,
                 **kwargs):
        """
        初始化车道级交通数据集
        
        Args:
            static_data_path: 静态道路数据文件路径(graph.json)
            dynamic_data_path: 动态交通数据文件路径(csv)
            mask_data_path: 用户自定义掩码文件路径(csv)，可选
            feature_cols: 要使用的特征列名列表，默认使用所有数值特征
            time_col: 动态数据中的时间列名
            lane_id_col: 车道ID列名
            mask_time_col: 掩码文件中的时间列名
            mask_lane_col: 掩码文件中的车道ID列名
            mask_value_col: 掩码文件中的观测值列名
            window_size: 时间窗口大小
            stride: 时间步长
            val_len: 验证集比例
            test_len: 测试集比例
            impute_nans: 是否填充缺失值
            fill_value: 缺失值填充值
        """
        super().__init__(**kwargs)
        
        self.static_data_path = static_data_path
        self.dynamic_data_path = dynamic_data_path
        self.mask_data_path = mask_data_path
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS
        self.time_col = time_col
        self.lane_id_col = lane_id_col
        self.mask_time_col = mask_time_col
        self.mask_lane_col = mask_lane_col
        self.mask_value_col = mask_value_col
        self.window_size = window_size
        self.stride = stride
        self.val_len = val_len
        self.test_len = test_len
        self.impute_nans = impute_nans
        self.fill_value = fill_value
        
        # 加载和预处理数据
        self._load_data()
        self._preprocess_data()
        
    def _load_data(self):
        """加载静态道路数据和动态交通数据"""
        # 1. 加载静态道路数据 (graph.json)
        static_path = Path(self.static_data_path)
        if static_path.suffix == '.json':
            with open(static_path, 'r', encoding='utf-8') as f:
                static_data = json.load(f)
            if 'nodes' in static_data:
                self.static_nodes = static_data['nodes']
            else:
                self.static_nodes = static_data
        else:
            raise ValueError(f"静态数据文件应为JSON格式: {static_path.suffix}")
            
        print(f"✅ 加载静态道路数据: {len(self.static_nodes)} 个节点")
        
        # 2. 加载动态交通数据 (csv)
        dynamic_path = Path(self.dynamic_data_path)
        if dynamic_path.suffix == '.csv':
            self.dynamic_df = pd.read_csv(dynamic_path)
        else:
            raise ValueError(f"动态数据文件应为CSV格式: {dynamic_path.suffix}")
            
        print(f"✅ 加载动态交通数据: {self.dynamic_df.shape[0]} 条记录")
        
        # 3. 验证数据一致性
        static_lane_ids = set(node[self.lane_id_col] for node in self.static_nodes)
        dynamic_lane_ids = set(self.dynamic_df[self.lane_id_col])
        
        if not dynamic_lane_ids.issubset(static_lane_ids):
            missing = dynamic_lane_ids - static_lane_ids
            print(f"⚠️ 警告: 动态数据中有 {len(missing)} 个 lane_id 在静态数据中不存在: {missing}")
        
        print(f"✅ 数据一致性验证通过")
        
    def _preprocess_data(self):
        """数据预处理"""
        # 按时间和lane_id排序
        self.dynamic_df = self.dynamic_df.sort_values([self.time_col, self.lane_id_col])
        
        # 从动态数据创建唯一的时间戳索引
        self.timestamps = np.sort(self.dynamic_df[self.time_col].unique())
        
        # 从静态数据创建唯一的lane_id索引
        self.lane_ids = np.array([node[self.lane_id_col] for node in self.static_nodes])
        self.lane_ids = np.sort(np.unique(self.lane_ids))
        
        print(f"时间步数: {len(self.timestamps)}")
        print(f"车道数: {len(self.lane_ids)}")
        
        # 检查并过滤有效的特征列
        available_cols = [col for col in self.feature_cols if col in self.dynamic_df.columns]
        if len(available_cols) < len(self.feature_cols):
            missing_cols = set(self.feature_cols) - set(available_cols)
            print(f"⚠️ 警告: 以下特征列不存在: {missing_cols}")
        self.feature_cols = available_cols
        print(f"使用特征列: {self.feature_cols}")
        
        # 构建时空数据矩阵
        self._build_spatiotemporal_matrix()
        
        # 构建图连接
        self._build_graph_connectivity()
        
        # 创建训练/评估掩码
        self._create_masks()
        
    def _build_spatiotemporal_matrix(self):
        """构建时空数据矩阵"""
        n_times = len(self.timestamps)
        n_lanes = len(self.lane_ids)
        n_features = len(self.feature_cols)
        
        # 初始化数据矩阵为NaN
        self.data = np.full((n_times, n_lanes, n_features), np.nan)
        
        # 创建lane_id到索引的映射
        lane_id_to_idx = {lid: idx for idx, lid in enumerate(self.lane_ids)}
        time_to_idx = {t: idx for idx, t in enumerate(self.timestamps)}
        
        # 填充数据
        for _, row in self.dynamic_df.iterrows():
            time_idx = time_to_idx.get(row[self.time_col])
            lane_idx = lane_id_to_idx.get(row[self.lane_id_col])
            
            if time_idx is not None and lane_idx is not None:
                for f_idx, col in enumerate(self.feature_cols):
                    if col in row and pd.notna(row[col]):
                        # 处理 -1.0 表示不适用的情况
                        val = row[col]
                        if val == -1.0:
                            val = np.nan  # 或者保留-1.0，取决于你的需求
                        self.data[time_idx, lane_idx, f_idx] = val
        
        # 处理缺失值
        nan_ratio = np.isnan(self.data).mean()
        print(f"原始缺失值比例: {nan_ratio:.3f}")
        
        if self.impute_nans:
            # 使用前向填充
            for i in range(1, n_times):
                mask = np.isnan(self.data[i])
                self.data[i][mask] = self.data[i-1][mask]
            # 剩余的NaN用fill_value填充
            self.data = np.nan_to_num(self.data, nan=self.fill_value)
                
        print(f"数据矩阵形状: {self.data.shape}")
        print(f"填充后缺失值比例: {np.isnan(self.data).mean():.3f}")
        
    def _build_graph_connectivity(self):
        """构建基于节点连接规则的图连接"""
        n_lanes = len(self.lane_ids)
        adj_matrix = np.zeros((n_lanes, n_lanes))
        
        # 创建lane_id到索引的映射
        lane_id_to_idx = {lid: idx for idx, lid in enumerate(self.lane_ids)}
        
        # 遍历静态节点，构建邻接矩阵
        for node in self.static_nodes:
            source_lane = node[self.lane_id_col]
            if source_lane not in lane_id_to_idx:
                continue
            source_idx = lane_id_to_idx[source_lane]
            
            # 获取节点连接信息
            connections = node.get('node_connections', {})
            if isinstance(connections, str):
                try:
                    connections = json.loads(connections)
                except:
                    connections = {}
            
            # 处理不同类型的连接
            for conn_type, targets in connections.items():
                if not isinstance(targets, list):
                    targets = [targets]
                
                for target_lane in targets:
                    if target_lane in lane_id_to_idx:
                        target_idx = lane_id_to_idx[target_lane]
                        
                        # 根据连接类型设置权重
                        if conn_type == 'direct':
                            weight = 1.0
                        elif conn_type == 'near':
                            weight = 0.5
                        elif conn_type == 'crossing':
                            weight = 0.3
                        else:
                            weight = 0.1
                        
                        # 添加双向连接
                        adj_matrix[source_idx, target_idx] = max(adj_matrix[source_idx, target_idx], weight)
                        adj_matrix[target_idx, source_idx] = max(adj_matrix[target_idx, source_idx], weight)
        
        self.adjacency = adj_matrix
        print(f"图连接矩阵形状: {self.adjacency.shape}")
        print(f"连接数: {np.sum(adj_matrix > 0) // 2}")
        
    def _create_masks(self):
        """创建训练/评估掩码"""
        n_times, n_lanes, n_features = self.data.shape
        
        if self.mask_data_path is not None:
            self._load_user_mask()
            print(f"✅ 使用用户自定义掩码")
            print(f"   已观测数据比例: {self.training_mask.mean():.3f}")
            print(f"   未观测数据比例: {self.eval_mask.mean():.3f}")
        else:
            # 默认：所有数据用于训练，随机选择20%用于评估
            self.training_mask = np.ones((n_times, n_lanes, n_features), dtype=bool)
            
            np.random.seed(42)
            eval_indices = np.random.choice(
                n_times * n_lanes * n_features,
                size=int(0.2 * n_times * n_lanes * n_features),
                replace=False
            )
            
            self.eval_mask = np.zeros((n_times, n_lanes, n_features), dtype=bool)
            eval_mask_flat = self.eval_mask.reshape(-1)
            eval_mask_flat[eval_indices] = True
            print(f"✅ 使用随机生成的掩码")
            
    def _load_user_mask(self):
        """从用户提供的CSV文件加载掩码数据"""
        n_times, n_lanes, n_features = self.data.shape
        mask_path = Path(self.mask_data_path)
        
        if not mask_path.exists():
            raise ValueError(f"掩码文件不存在: {self.mask_data_path}")
        
        # 加载CSV掩码
        mask_df = pd.read_csv(mask_path)
        
        # 检查必需列
        required_cols = [self.mask_time_col, self.mask_lane_col, self.mask_value_col]
        missing_cols = [col for col in required_cols if col not in mask_df.columns]
        if missing_cols:
            raise ValueError(f"掩码文件缺少必需列: {missing_cols}")
        
        # 初始化掩码矩阵（默认所有位置都是未观测的）
        self.training_mask = np.zeros((n_times, n_lanes, n_features), dtype=bool)
        
        # 创建索引映射
        lane_id_to_idx = {lid: idx for idx, lid in enumerate(self.lane_ids)}
        time_to_idx = {t: idx for idx, t in enumerate(self.timestamps)}
        
        # 填充掩码
        for _, row in mask_df.iterrows():
            time_val = row[self.mask_time_col]
            lane_id = row[self.mask_lane_col]
            is_observed = bool(row[self.mask_value_col])
            
            time_idx = time_to_idx.get(time_val)
            lane_idx = lane_id_to_idx.get(lane_id)
            
            if time_idx is not None and lane_idx is not None:
                # 对所有特征都使用相同的掩码
                self.training_mask[time_idx, lane_idx, :] = is_observed
        
        # 评估掩码是训练掩码的反
        self.eval_mask = ~self.training_mask
        
    def get_connectivity(self, threshold: float = 0.1, 
                        include_self: bool = False,
                        force_symmetric: bool = True) -> np.ndarray:
        """获取图连接矩阵"""
        adj = self.adjacency.copy()
        
        # 应用阈值
        adj[adj < threshold] = 0
        
        if not include_self:
            np.fill_diagonal(adj, 0)
            
        if force_symmetric:
            adj = (adj + adj.T) / 2
            
        return adj
        
    def numpy(self, return_idx: bool = False) -> Union[Tuple, np.ndarray]:
        """返回numpy格式的数据"""
        if return_idx:
            return self.data, self.timestamps, self.lane_ids
        return self.data
        
    def datetime_encoded(self, encoding: List[str]) -> pd.DataFrame:
        """获取时间编码"""
        # 对于 start_frame 格式的时间，创建简单的周期编码
        n_times = len(self.timestamps)
        df = pd.DataFrame({'frame': self.timestamps})
        
        if 'day' in encoding:
            # 假设每天有固定数量的帧，这里简化处理
            frames_per_day = 24 * 6  # 假设每10分钟一帧，每天144帧
            df['day'] = (df['frame'] // frames_per_day) % 7
        if 'week' in encoding:
            frames_per_week = 24 * 6 * 7
            df['week'] = (df['frame'] // frames_per_week) % 52
        if 'hour' in encoding:
            frames_per_hour = 6  # 假设每10分钟一帧
            df['hour'] = (df['frame'] // frames_per_hour) % 24
            
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
        return len(self.lane_ids)
        
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
            'lane_ids': self.lane_ids
        }
