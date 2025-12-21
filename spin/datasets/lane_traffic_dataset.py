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
from ..layers import DTStateFilter


class LaneTrafficDataset(Dataset):
    """
    车道级交通状况数据集
    
    数据格式：
    - 静态数据(graph.json): 包含 lane_id 和 node_connections
    - 动态数据(csv): 包含 lane_id, start_frame, avg_speed, avg_occupancy 等特征
    - 掩码数据(csv): 包含 start_frame, lane_id, is_observed
    
    支持两种输入方式：
    1. 单组数据：直接传入 static_data_path, dynamic_data_path, mask_data_path
    2. 多组数据：传入 data_groups 列表，每组包含 static, dynamic, mask 路径
    """
    
    # 默认特征列（可配置）
    DEFAULT_FEATURE_COLS = [
        'avg_speed', 'avg_occupancy', 'total_vehicles'
    ]
    
    def __init__(self, 
                 static_data_path: Optional[str] = None,
                 dynamic_data_path: Optional[str] = None,
                 mask_data_path: Optional[str] = None,
                 data_groups: Optional[List[Dict[str, str]]] = None,
                 feature_cols: Optional[List[str]] = None,
                 time_col: str = 'start_frame',
                 lane_id_col: str = 'lane_id',
                 mask_time_col: str = 'start_frame',
                 mask_lane_col: str = 'lane_id',
                 mask_value_col: str = 'is_observed',
                 window_size: int = 10,
                 stride: int = 1,
                 val_len: float = 0.1,
                 test_len: float = 0.2,
                 impute_nans: bool = True,
                 fill_value: float = 0.0, # 缺失值填充值
                 enable_dtsf: bool = True, # 是否启用双时间尺度拥堵隐状态
                 dtsf_gamma: float = 0.7, # 双时间尺度拥堵隐状态的gamma参数，越接近 1，历史状态占比越大，曲线越平滑、反应越慢
                 dtsf_delta: float = 5.0, # 双时间尺度拥堵隐状态的delta参数
                 dtsf_vth_ratio: float = 0.8, # 双时间尺度拥堵隐状态的vth_ratio参数，当速度低于基础速度的这个值左右就开始被认为是拥堵。
                 dtsf_initial_z: float = 1.0, # 双时间尺度拥堵隐状态的初始拥堵状态参数
                 dtsf_v_base_init: float = 45.0, # 双时间尺度拥堵隐状态的v_base_init参数，基础速度的初始值
                 dtsf_no_car_value: Optional[float] = None,
                 dtsf_auto_no_car: bool = True, # 是否自动识别"无车"标记值
                 dtsf_treat_no_car_as_missing: bool = True, # 是否将"无车"标记值视为缺失值
                 dtsf_no_car_eps: float = 1e-3, # "无车"标记值的epsilon参数
                 dtsf_device: str = 'cuda', # 双时间尺度拥堵隐状态的设备参数
                 **kwargs):
        """
        初始化车道级交通数据集
        
        Args:
            static_data_path: 静态道路数据文件路径(graph.json)，单组数据时使用
            dynamic_data_path: 动态交通数据文件路径(csv)，单组数据时使用
            mask_data_path: 用户自定义掩码文件路径(csv)，可选
            data_groups: 多组数据配置列表，每组格式为:
                         [{"static": "path1.json", "dynamic": "path1.csv", "mask": "mask1.csv"}, ...]
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
        
        # 处理数据路径：支持单组或多组
        if data_groups is not None:
            self.data_groups = data_groups
        elif static_data_path is not None and dynamic_data_path is not None:
            self.data_groups = [{
                'static': static_data_path,
                'dynamic': dynamic_data_path,
                'mask': mask_data_path
            }]
        else:
            raise ValueError("必须提供 data_groups 或 (static_data_path + dynamic_data_path)")
        
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
        self.enable_dtsf = enable_dtsf
        self.dtsf_gamma = dtsf_gamma
        self.dtsf_delta = dtsf_delta
        self.dtsf_vth_ratio = dtsf_vth_ratio
        self.dtsf_initial_z = dtsf_initial_z
        self.dtsf_v_base_init = dtsf_v_base_init
        self.dtsf_no_car_value = dtsf_no_car_value
        self.dtsf_auto_no_car = dtsf_auto_no_car
        self.dtsf_treat_no_car_as_missing = dtsf_treat_no_car_as_missing
        self.dtsf_no_car_eps = dtsf_no_car_eps
        self.dtsf_device = dtsf_device
        
        # 保存归一化参数（用于推理时反归一化）
        self.speed_normalization_params = None  # {'speed_min': float, 'speed_max': float, 'is_normalized': bool}
        
        # 保存用于训练时循环选择的mask文件列表（从data_groups中自动提取）
        self.mask_files = []  # 实际使用的mask文件列表（包含匹配信息）
        self.current_mask_file = None  # 当前使用的mask文件路径
        self.current_mask_index = 0  # 当前选择的mask文件索引（用于循环选择）
        
        # 加载和预处理数据
        self._load_data()
        self._preprocess_data()
        
        # 初始化mask_files列表（如果未指定，从data_groups中提取）
        self._initialize_mask_files()
        
    def _load_data(self):
        """加载静态道路数据和动态交通数据（支持多组）"""
        self.static_nodes = []
        self.dynamic_df = pd.DataFrame()
        self.mask_data_paths = []  # 保存所有mask路径供后续使用
        self.dynamic_file_info = []  # 保存每个dynamic文件的信息（用于匹配mask文件）
        
        # 用于记录时间戳偏移量，避免不同文件的时间戳冲突
        time_offset = 0.0
        max_timestamp_so_far = None
        
        for i, group in enumerate(self.data_groups):
            print(f"\n📂 加载第 {i+1}/{len(self.data_groups)} 组数据...")
            
            # 1. 加载静态道路数据 (graph.json)
            static_path = Path(group['static'])
            if static_path.suffix == '.json':
                with open(static_path, 'r', encoding='utf-8') as f:
                    static_data = json.load(f)
                if 'nodes' in static_data:
                    nodes = static_data['nodes']
                else:
                    nodes = static_data
                self.static_nodes.extend(nodes)
                print(f"   ✅ 静态数据: {len(nodes)} 个节点")
            else:
                raise ValueError(f"静态数据文件应为JSON格式: {static_path.suffix}")
            
            # 2. 加载动态交通数据 (csv)
            dynamic_path = Path(group['dynamic'])
            if dynamic_path.suffix == '.csv':
                df = pd.read_csv(dynamic_path)
                
                # 检查时间戳列是否存在
                if self.time_col not in df.columns:
                    raise ValueError(f"动态数据文件 {dynamic_path} 缺少时间列: {self.time_col}")
                
                # 获取当前文件的时间戳范围
                current_times = df[self.time_col].values
                current_min_time = np.min(current_times)
                current_max_time = np.max(current_times)
                time_span = current_max_time - current_min_time
                
                # 初始化当前文件的偏移量（第一个文件为0）
                current_file_offset = 0.0
                
                # 如果这不是第一个文件，且时间戳有重叠风险，则添加偏移量
                if i > 0 and max_timestamp_so_far is not None:
                    # 计算偏移量：之前最大时间戳 + 时间间隔 + 1（确保不重叠）
                    # 时间间隔取当前文件的时间跨度，或者如果无法确定则使用一个较大的值
                    if time_span > 0:
                        # 使用当前文件的时间跨度作为间隔
                        time_gap = time_span * 0.1  # 添加10%的间隔作为缓冲
                    else:
                        # 如果时间跨度为0（所有时间戳相同），使用一个固定间隔
                        time_gap = 1.0
                    
                    current_file_offset = max_timestamp_so_far + time_gap + 1.0
                    print(f"   ⏰ 检测到时间戳冲突风险，为文件 {i+1} 添加时间偏移量: {current_file_offset:.2f}")
                
                # 应用时间偏移量
                df[self.time_col] = df[self.time_col] + current_file_offset
                
                # 更新最大时间戳
                current_max_time_adjusted = current_max_time + current_file_offset
                if max_timestamp_so_far is None:
                    max_timestamp_so_far = current_max_time_adjusted
                else:
                    max_timestamp_so_far = max(max_timestamp_so_far, current_max_time_adjusted)
                
                self.dynamic_df = pd.concat([self.dynamic_df, df], ignore_index=True)
                print(f"   ✅ 动态数据: {df.shape[0]} 条记录 (时间范围: {current_min_time + current_file_offset:.2f} ~ {current_max_time_adjusted:.2f})")
            else:
                raise ValueError(f"动态数据文件应为CSV格式: {dynamic_path.suffix}")
            
            # 3. 保存dynamic文件信息（用于匹配mask文件）
            # 同时记录该文件的时间戳范围（已应用偏移量），用于后续确定文件边界
            file_min_time = current_min_time + current_file_offset
            file_max_time = current_max_time + current_file_offset
            
            # 支持mask字段是单个文件路径或文件路径列表
            mask_path_or_list = group.get('mask')
            if mask_path_or_list is None:
                mask_paths = []
            elif isinstance(mask_path_or_list, list):
                mask_paths = mask_path_or_list
            else:
                mask_paths = [mask_path_or_list]
            
            self.dynamic_file_info.append({
                'dynamic_path': str(dynamic_path),
                'time_offset': current_file_offset,
                'mask_paths': mask_paths,  # 改为列表，支持多个mask文件
                'time_range': (file_min_time, file_max_time)  # 记录时间戳范围
            })
            
            # 4. 保存mask路径（同时保存对应的时间偏移量信息）
            # 为了向后兼容，仍然保存到mask_data_paths（但只保存第一个，如果有的话）
            if mask_paths:
                self.mask_data_paths.append({
                    'path': mask_paths[0],  # 向后兼容：只保存第一个
                    'time_offset': current_file_offset
                })
        
        print(f"\n📊 合并后总计:")
        print(f"   静态节点: {len(self.static_nodes)} 个")
        print(f"   动态记录: {self.dynamic_df.shape[0]} 条")
        print(f"   时间戳范围: {self.dynamic_df[self.time_col].min():.2f} ~ {self.dynamic_df[self.time_col].max():.2f}")
        
        # 4. 验证数据一致性
        static_lane_ids = set(node[self.lane_id_col] for node in self.static_nodes)
        dynamic_lane_ids = set(self.dynamic_df[self.lane_id_col])
        
        if not dynamic_lane_ids.issubset(static_lane_ids):
            missing = dynamic_lane_ids - static_lane_ids
            print(f"⚠️ 警告: 动态数据中有 {len(missing)} 个 lane_id 在静态数据中不存在")
        
        print(f"✅ 数据一致性验证通过")
        
    def _preprocess_data(self):
        """数据预处理"""
        # 按时间和lane_id排序
        self.dynamic_df = self.dynamic_df.sort_values([self.time_col, self.lane_id_col])
        
        # 从动态数据创建唯一的时间戳索引
        self.timestamps = np.sort(self.dynamic_df[self.time_col].unique())
        # 如果提供了mask文件，将其中的时间戳并入时间轴，确保mask与数据时间对齐
        # 从dynamic_file_info中读取所有mask文件
        has_mask_files = False
        for dyn_info in self.dynamic_file_info:
            mask_paths = dyn_info.get('mask_paths', [])
            if not mask_paths:
                mask_path = dyn_info.get('mask_path')
                if mask_path is not None:
                    mask_paths = [mask_path]
            if mask_paths:
                has_mask_files = True
                break
        
        if has_mask_files:
            mask_times = []
            for dyn_info in self.dynamic_file_info:
                mask_paths = dyn_info.get('mask_paths', [])
                if not mask_paths:
                    # 向后兼容
                    mask_path = dyn_info.get('mask_path')
                    if mask_path is not None:
                        mask_paths = [mask_path]
                
                time_offset = dyn_info.get('time_offset', 0.0)
                
                for mask_path in mask_paths:
                    if mask_path is None:
                        continue
                    mp = Path(mask_path)
                    if not mp.exists():
                        continue
                    try:
                        mask_df = pd.read_csv(mp)
                        if self.mask_time_col in mask_df.columns:
                            # 应用相同的时间偏移量
                            mask_times_adjusted = mask_df[self.mask_time_col].values + time_offset
                            mask_times.extend(mask_times_adjusted.tolist())
                    except Exception as e:
                        print(f"⚠️ 警告: 读取掩码文件时间列失败 {mp}: {e}")
            if mask_times:
                union_times = np.unique(np.concatenate([self.timestamps, np.array(mask_times)]))
                if len(union_times) != len(self.timestamps):
                    added = len(union_times) - len(self.timestamps)
                    print(f"✅ 已将掩码文件中的 {added} 个时间戳并入时间轴，保证与mask对齐")
                self.timestamps = union_times
        
        # 记录每个文件对应的时间索引范围（用于避免窗口跨越文件边界）
        self.file_boundaries = []  # 每个元素是 (start_idx, end_idx) 表示文件在时间索引中的范围
        time_to_idx = {t: idx for idx, t in enumerate(self.timestamps)}
        
        for i, dyn_info in enumerate(self.dynamic_file_info):
            dynamic_path = Path(dyn_info['dynamic_path'])
            
            # 获取该文件的时间戳范围（已应用偏移量）
            if 'time_range' in dyn_info:
                file_min_time, file_max_time = dyn_info['time_range']
            else:
                # 向后兼容：如果没有 time_range，尝试从 dynamic_df 中推断
                # 这需要重新读取文件，效率较低，但可以工作
                try:
                    df_original = pd.read_csv(dynamic_path)
                    time_offset = dyn_info['time_offset']
                    original_times = df_original[self.time_col].values
                    adjusted_times = original_times + time_offset
                    file_min_time = np.min(adjusted_times)
                    file_max_time = np.max(adjusted_times)
                except:
                    # 如果无法读取，跳过这个文件
                    self.file_boundaries.append((0, 0))
                    continue
            
            # 找到该文件时间戳范围内的所有时间索引
            # 注意：由于 mask 文件可能添加了额外时间戳，我们需要检查所有在范围内的索引
            valid_indices = []
            for t_idx, timestamp in enumerate(self.timestamps):
                # 检查时间戳是否在该文件的范围内（允许小的浮点误差）
                if file_min_time - 1e-6 <= timestamp <= file_max_time + 1e-6:
                    valid_indices.append(t_idx)
            
            if valid_indices:
                start_idx = min(valid_indices)
                end_idx = max(valid_indices) + 1  # end_idx 是开区间
                self.file_boundaries.append((start_idx, end_idx))
                print(f"   文件 {i+1} ({dynamic_path.name}) 时间索引范围: [{start_idx}, {end_idx})")
            else:
                # 如果没有找到有效索引，使用空范围
                self.file_boundaries.append((0, 0))
                print(f"   ⚠️ 警告: 文件 {i+1} ({dynamic_path.name}) 未找到有效时间索引")
        
        if len(self.file_boundaries) > 1:
            print(f"✅ 已记录 {len(self.file_boundaries)} 个文件的边界信息，用于防止窗口跨越文件边界")
        
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

        # 在填充缺失值前先基于原始速度序列构造 DTSF 拥堵状态
        if self.enable_dtsf:
            self._append_dtsf_state()
        
        if self.impute_nans:
            # 使用前向填充
            for i in range(1, n_times):
                mask = np.isnan(self.data[i])
                self.data[i][mask] = self.data[i-1][mask]
            # 剩余的NaN用fill_value填充
            self.data = np.nan_to_num(self.data, nan=self.fill_value)
                
        print(f"数据矩阵形状: {self.data.shape}")
        print(f"填充后缺失值比例: {np.isnan(self.data).mean():.3f}")

    def _append_dtsf_state(self):
        """基于 avg_speed 计算双时间尺度拥堵隐状态并追加为新特征
        
        注意：如果 avg_speed 是绝对速度值（km/h），在计算 DTSF 后会自动归一化到 0-1 范围
        以与其他归一化特征保持一致的尺度
        """
        if 'avg_speed' not in self.feature_cols:
            print("⚠️ DTSF 跳过：未找到 avg_speed 特征列")
            return

        speed_idx = self.feature_cols.index('avg_speed')
        speed_matrix = self.data[..., speed_idx].copy()  # 保存原始速度值用于 DTSF 计算

        if np.isnan(speed_matrix).all():
            print("⚠️ DTSF 跳过：avg_speed 全为缺失")
            return

        # 检测速度值是否已经是归一化的（0-1范围）还是绝对速度值（km/h）
        finite_vals = speed_matrix[~np.isnan(speed_matrix)]
        if finite_vals.size > 0:
            speed_max = np.nanmax(speed_matrix)
            speed_min = np.nanmin(speed_matrix)
            is_normalized = speed_max <= 1.5  # 如果最大值小于等于1.5，认为是归一化的
        else:
            is_normalized = True  # 如果全为NaN，默认认为是归一化的
            speed_max = 1.0
            speed_min = 0.0

        # 自动识别"无车"标记值（针对 0~1 归一化且无车=1 的场景）
        no_car_value = self.dtsf_no_car_value
        if self.dtsf_auto_no_car and no_car_value is None:
            if is_normalized:
                no_car_value = 1.0

        z_state = np.zeros_like(speed_matrix, dtype=np.float32)
        n_times, n_lanes = speed_matrix.shape

        for lane_idx in range(n_lanes):
            lane_speed = speed_matrix[:, lane_idx]
            valid_vals = lane_speed[~np.isnan(lane_speed)]
            v_base_init = float(valid_vals[0]) if valid_vals.size > 0 else self.dtsf_v_base_init

            filter_module = DTStateFilter(
                gamma=self.dtsf_gamma,
                delta=self.dtsf_delta,
                vth_ratio=self.dtsf_vth_ratio,
                v_base_init=v_base_init,
                initial_z=self.dtsf_initial_z,
                device=self.dtsf_device,
            )

            with torch.no_grad():
                for t, v in enumerate(lane_speed):
                    v_obs = None
                    if not np.isnan(v):
                        is_no_car = False
                        if self.dtsf_treat_no_car_as_missing and no_car_value is not None:
                            is_no_car = v >= (no_car_value - self.dtsf_no_car_eps)
                        if not is_no_car:
                            v_obs = float(v)
                    z_val = filter_module(v_obs)
                    z_state[t, lane_idx] = float(z_val.detach().cpu())

        # 添加 DTSF 状态特征
        self.data = np.concatenate([self.data, z_state[..., None]], axis=-1)
        self.feature_cols.append('dtsf_congestion')
        
        # 如果 avg_speed 是绝对速度值（非归一化），需要归一化到 0-1 范围
        # 以与其他归一化特征保持一致的尺度，避免 StandardScaler 标准化时尺度差异过大
        if not is_normalized and finite_vals.size > 0:
            # 使用 min-max 归一化：将速度值归一化到 [0, 1] 范围
            # 使用全局的最大最小值，确保所有车道使用相同的归一化参数
            speed_range = speed_max - speed_min
            if speed_range > 1e-6:  # 避免除零
                normalized_speed = (speed_matrix - speed_min) / speed_range
                # 更新数据矩阵中的 avg_speed 值
                self.data[..., speed_idx] = normalized_speed
                # 保存归一化参数，用于推理时反归一化
                self.speed_normalization_params = {
                    'speed_min': float(speed_min),
                    'speed_max': float(speed_max),
                    'is_normalized': False,
                    'feature_idx': speed_idx
                }
                print(f"✅ 已将 avg_speed 从绝对速度值 ({speed_min:.2f}-{speed_max:.2f} km/h) 归一化到 [0, 1] 范围")
                print(f"   已保存归一化参数: min={speed_min:.2f}, max={speed_max:.2f} km/h")
            else:
                print(f"⚠️ avg_speed 值范围过小 ({speed_min:.2f}-{speed_max:.2f})，跳过归一化")
        else:
            # 如果已经是归一化的，也保存参数（虽然不需要反归一化）
            self.speed_normalization_params = {
                'speed_min': 0.0,
                'speed_max': 1.0,
                'is_normalized': True,
                'feature_idx': speed_idx
            }
        
        print("✅ 已添加 DTSF 拥堵状态特征，当前特征数:", len(self.feature_cols))
        
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
                            weight = 0.5
                        else:
                            weight = 0.5
                        
                        # 添加双向连接
                        adj_matrix[source_idx, target_idx] = max(adj_matrix[source_idx, target_idx], weight)
                        adj_matrix[target_idx, source_idx] = max(adj_matrix[target_idx, source_idx], weight)
        
        self.adjacency = adj_matrix
        print(f"图连接矩阵形状: {self.adjacency.shape}")
        print(f"连接数: {np.sum(adj_matrix > 0) // 2}")
        
    def _create_masks(self):
        """创建训练/评估掩码（支持多组mask文件，每个mask可以是列表）"""
        n_times, n_lanes, n_features = self.data.shape
        
        # 检查是否有任何mask文件（从dynamic_file_info中检查）
        has_masks = False
        for dyn_info in self.dynamic_file_info:
            mask_paths = dyn_info.get('mask_paths', [])
            if not mask_paths:
                # 向后兼容
                mask_path = dyn_info.get('mask_path')
                if mask_path is not None:
                    mask_paths = [mask_path]
            if mask_paths:
                has_masks = True
                break
        
        if has_masks:
            self._load_user_masks()
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
            
    def _load_user_masks(self):
        """从用户提供的多个CSV文件加载掩码数据（支持每个dynamic文件对应多个mask文件）"""
        n_times, n_lanes, n_features = self.data.shape
        
        # 初始化掩码矩阵（默认所有位置都是未观测的）
        self.training_mask = np.zeros((n_times, n_lanes, n_features), dtype=bool)
        
        # 创建索引映射
        lane_id_to_idx = {lid: idx for idx, lid in enumerate(self.lane_ids)}
        time_to_idx = {t: idx for idx, t in enumerate(self.timestamps)}
        
        # 从dynamic_file_info中加载所有mask文件
        file_idx = 0
        for dyn_info in self.dynamic_file_info:
            mask_paths = dyn_info.get('mask_paths', [])
            if not mask_paths:
                # 向后兼容：如果没有mask_paths，尝试使用mask_path
                mask_path = dyn_info.get('mask_path')
                if mask_path is not None:
                    mask_paths = [mask_path]
            
            time_offset = dyn_info.get('time_offset', 0.0)
            
            # 加载该dynamic文件对应的所有mask文件
            for mask_path in mask_paths:
                if mask_path is None:
                    continue
                
                file_idx += 1
                mask_path_obj = Path(mask_path)
                if not mask_path_obj.exists():
                    print(f"⚠️ 警告: 掩码文件不存在，跳过: {mask_path_obj}")
                    continue
                
                print(f"   加载掩码文件 {file_idx}: {mask_path_obj.name} (时间偏移量: {time_offset:.2f})")
                try:
                    mask_df = pd.read_csv(mask_path_obj)
                    
                    # 检查必需列
                    required_cols = [self.mask_time_col, self.mask_lane_col, self.mask_value_col]
                    missing_cols = [col for col in required_cols if col not in mask_df.columns]
                    if missing_cols:
                        raise ValueError(f"掩码文件 {mask_path_obj} 缺少必需列: {missing_cols}")
                    
                    # 应用时间偏移量到mask文件的时间戳
                    mask_df = mask_df.copy()
                    mask_df[self.mask_time_col] = mask_df[self.mask_time_col] + time_offset
                    
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
                        elif time_idx is None:
                            # 时间戳不在时间轴中，可能是mask文件的时间戳范围超出了数据范围
                            pass  # 静默忽略，因为时间戳可能已经在合并时处理过了
                except Exception as e:
                    print(f"⚠️ 警告: 加载掩码文件失败 {mask_path_obj}: {e}")
                    continue
        
        # 评估掩码是训练掩码的反
        self.eval_mask = ~self.training_mask
    
    def _initialize_mask_files(self):
        """
        从data_groups中自动提取所有mask文件
        支持每个dynamic文件对应多个mask文件（mask字段可以是列表）
        """
        print(f"\n📋 从data_groups中自动提取mask文件...")
        for dyn_info in self.dynamic_file_info:
            mask_paths = dyn_info.get('mask_paths', [])
            if not mask_paths:
                # 向后兼容：如果没有mask_paths，尝试使用mask_path
                mask_path = dyn_info.get('mask_path')
                if mask_path is not None:
                    mask_paths = [mask_path]
            
            # 支持每个dynamic文件对应多个mask文件
            for mask_path in mask_paths:
                if mask_path is None:
                    continue
                mask_path_obj = Path(mask_path)
                if mask_path_obj.exists():
                    self.mask_files.append({
                        'path': str(mask_path_obj),
                        'time_offset': dyn_info['time_offset'],
                        'dynamic_path': dyn_info['dynamic_path']
                    })
                    print(f"   ✅ {mask_path_obj.name} -> {Path(dyn_info['dynamic_path']).name} (时间偏移: {dyn_info['time_offset']:.2f})")
                else:
                    print(f"   ⚠️  mask文件不存在，跳过: {mask_path_obj}")
        
        if len(self.mask_files) == 0:
            print(f"⚠️  警告: 没有可用的mask文件用于动态切换")
        else:
            print(f"✅ 共找到 {len(self.mask_files)} 个mask文件可用于动态切换")
    
    def switch_mask_sequentially(self, epoch: Optional[int] = None) -> bool:
        """
        从mask_files列表中按顺序循环选择一个mask文件并加载，用于训练时动态切换mask
        自动应用对应dynamic文件的时间偏移量
        
        Args:
            epoch: 当前epoch编号，如果为None则使用内部索引自动递增
            
        Returns:
            bool: 是否成功切换mask
        """
        if not self.mask_files:
            return False
        
        # 如果提供了epoch编号，使用它来选择mask文件（循环）
        if epoch is not None:
            mask_index = epoch % len(self.mask_files)
        else:
            # 否则使用内部索引，并在每次调用后递增
            mask_index = self.current_mask_index
            self.current_mask_index = (self.current_mask_index + 1) % len(self.mask_files)
        
        # 按顺序选择一个mask文件（包含匹配信息）
        selected_mask_info = self.mask_files[mask_index]
        selected_mask_file = selected_mask_info['path']
        time_offset = selected_mask_info['time_offset']
        dynamic_path = selected_mask_info['dynamic_path']
        
        self.current_mask_file = selected_mask_file
        
        print(f"🔄 切换到mask文件 ({mask_index + 1}/{len(self.mask_files)}): {Path(selected_mask_file).name}")
        print(f"   对应dynamic文件: {Path(dynamic_path).name}")
        print(f"   时间偏移量: {time_offset:.2f}")
        
        # 加载选中的mask文件
        n_times, n_lanes, n_features = self.data.shape
        
        # 初始化掩码矩阵（默认所有位置都是未观测的）
        new_training_mask = np.zeros((n_times, n_lanes, n_features), dtype=bool)
        
        # 创建索引映射
        lane_id_to_idx = {lid: idx for idx, lid in enumerate(self.lane_ids)}
        time_to_idx = {t: idx for idx, t in enumerate(self.timestamps)}
        
        # 加载mask文件
        mask_path = Path(selected_mask_file)
        if not mask_path.exists():
            print(f"⚠️ 警告: 掩码文件不存在，跳过: {mask_path}")
            return False
        
        try:
            mask_df = pd.read_csv(mask_path)
            
            # 检查必需列
            required_cols = [self.mask_time_col, self.mask_lane_col, self.mask_value_col]
            missing_cols = [col for col in required_cols if col not in mask_df.columns]
            if missing_cols:
                print(f"⚠️ 警告: 掩码文件 {mask_path} 缺少必需列: {missing_cols}")
                return False
            
            # 应用时间偏移量到mask文件的时间戳（确保与对应的dynamic文件对齐）
            mask_df = mask_df.copy()
            mask_df[self.mask_time_col] = mask_df[self.mask_time_col] + time_offset
            
            # 填充掩码
            for _, row in mask_df.iterrows():
                time_val = row[self.mask_time_col]
                lane_id = row[self.mask_lane_col]
                is_observed = bool(row[self.mask_value_col])
                
                time_idx = time_to_idx.get(time_val)
                lane_idx = lane_id_to_idx.get(lane_id)
                
                if time_idx is not None and lane_idx is not None:
                    # 对所有特征都使用相同的掩码
                    new_training_mask[time_idx, lane_idx, :] = is_observed
            
            # 更新mask
            self.training_mask = new_training_mask
            self.eval_mask = ~self.training_mask
            
            print(f"✅ 已更新mask，已观测数据比例: {self.training_mask.mean():.3f}")
            return True
            
        except Exception as e:
            print(f"⚠️ 警告: 加载掩码文件失败: {e}")
            return False
        
    def get_connectivity(self, threshold: float = 0.1, 
                        include_self: bool = False,
                        force_symmetric: bool = True) -> np.ndarray:
        """获取图连接矩阵"""
        adj = self.adjacency.copy()
        
        # 应用阈值，转为二值矩阵
        adj = (adj >= threshold).astype(np.uint8)
        
        if not include_self:
            np.fill_diagonal(adj, 0)
            
        if force_symmetric:
            adj = np.maximum(adj, adj.T)
            
        return adj
        
    def numpy(self, return_idx: bool = False) -> Union[Tuple, np.ndarray]:
        """返回numpy格式的数据"""
        if return_idx:
            return self.data, self.timestamps, self.lane_ids
        return self.data
        
    def datetime_encoded(self, encoding: List[str]) -> pd.DataFrame:
        """
        获取时间编码 - 使用真实时间差（不做 sin/cos），并保留相对进度

        返回两列：
        - time_linear: 相对进度 [0,1]
        - delta_t: 相邻时间步的真实时间差（与原始时间戳同单位），首个时间步置 0
        """
        n_times = len(self.timestamps)
        df = pd.DataFrame(index=range(n_times))

        # 归一化时间位置 [0, 1]
        t_min, t_max = self.timestamps.min(), self.timestamps.max()
        normalized_t = (self.timestamps - t_min) / (t_max - t_min + 1e-8)
        df['time_linear'] = normalized_t

        # 真实时间差特征（首步为 0）
        ts = self.timestamps.astype(float)
        delta = np.diff(ts, prepend=ts[0])
        df['delta_t'] = delta

        return df
        
    def get_splitter(self, val_len: float = None, test_len: float = None):
        """获取数据分割器"""
        from tsl.data.datamodule.splitters import TemporalSplitter
        
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
