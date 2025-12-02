import copy
import os
import sys

# 将项目根目录添加到 Python 路径，以便导入 spin 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import tsl
import yaml
from pathlib import Path
from tsl import config
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import AirQuality, MetrLA, PemsBay
from tsl.imputers import Imputer
from tsl.nn.models.imputation import GRINModel
from tsl.ops.imputation import add_missing_values, sample_mask
from tsl.utils import ArgParser, parser_utils, numpy_metrics
from tsl.utils.python_utils import ensure_list

from spin.baselines import SAITS, TransformerModel, BRITS
from spin.imputers import SPINImputer, SAITSImputer, BRITSImputer
from spin.models import SPINModel, SPINHierarchicalModel
from spin.datasets.lane_traffic_dataset import LaneTrafficDataset


def get_model_classes(model_str):
    if model_str == 'spin':
        model, filler = SPINModel, SPINImputer
    elif model_str == 'spin_h':
        model, filler = SPINHierarchicalModel, SPINImputer
    elif model_str == 'grin':
        model, filler = GRINModel, Imputer
    elif model_str == 'saits':
        model, filler = SAITS, SAITSImputer
    elif model_str == 'transformer':
        model, filler = TransformerModel, SPINImputer
    elif model_str == 'brits':
        model, filler = BRITS, BRITSImputer
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name: str, data_path: str = None, 
                static_data_path: str = None, mask_data_path: str = None,
                feature_cols: list = None):
    """
    获取数据集
    
    Args:
        dataset_name: 数据集名称
        data_path: 动态交通数据路径 (用于lane数据集)
        static_data_path: 静态道路数据路径 (用于lane数据集)
        mask_data_path: 掩码文件路径 (用于lane数据集，可选)
        feature_cols: 特征列名列表 (用于lane数据集)
    """
    # 支持车道级交通数据集
    if dataset_name == 'lane':
        if static_data_path is not None and data_path is not None:
            return LaneTrafficDataset(
                static_data_path=static_data_path,
                dynamic_data_path=data_path,
                mask_data_path=mask_data_path,
                feature_cols=feature_cols,
                impute_nans=True
            )
        else:
            raise ValueError("lane数据集需要指定 --static-data-path 和 --data-path")
    
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    # build missing dataset
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")
    if dataset_name == 'la':
        return add_missing_values(MetrLA(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    raise ValueError(f"Invalid dataset name: {dataset_name}.")


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--config", type=str, default='inference.yaml')
    parser.add_argument("--root", type=str, default='log')
    
    # Lane dataset params
    parser.add_argument("--checkpoint-path", type=str, default=None,
                       help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to dynamic traffic data file (csv) for lane dataset")
    parser.add_argument("--static-data-path", type=str, default=None,
                       help="Path to static road data file (graph.json) for lane dataset")
    parser.add_argument("--mask-data-path", type=str, default=None,
                       help="Path to mask data file (csv) for lane dataset")
    parser.add_argument("--feature-cols", type=str, default=None,
                       help="Comma-separated feature column names for lane dataset")

    # Data sparsity params
    parser.add_argument('--p-fault', type=float, default=0.0)
    parser.add_argument('--p-noise', type=float, default=0.75)
    parser.add_argument('--test-mask-seed', type=int, default=None)

    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=32)

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    
    # Output params
    parser.add_argument("--output-path", type=str, default=None,
                       help="Path to save imputed results (csv file)")

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as fp:
                config_args = yaml.load(fp, Loader=yaml.FullLoader)
            for arg in config_args:
                setattr(args, arg, config_args[arg])

    return args


def load_model(exp_dir, exp_config, dm, checkpoint_path=None, u_size=None):
    model_cls, imputer_class = get_model_classes(exp_config['model_name'])
    
    # 如果没有提供u_size，从config中读取，或者使用默认值
    if u_size is None:
        u_size = exp_config.get('u_size', 1)
    
    additional_model_hparams = dict(n_nodes=dm.n_nodes,
                                    input_size=dm.n_channels,
                                    u_size=u_size,
                                    output_size=dm.n_channels,
                                    window_size=dm.window)

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**exp_config, **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(exp_config,
                                                       imputer_class,
                                                       return_dict=True)
    
    # 如果提供了checkpoint_path，直接从checkpoint加载
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"从checkpoint加载模型: {checkpoint_path}")
        # 从checkpoint加载时，需要提供所有必要的参数
        imputer = imputer_class.load_from_checkpoint(
            checkpoint_path,
            model_class=model_cls,
            model_kwargs=model_kwargs,
            optim_class=torch.optim.Adam,
            optim_kwargs={},
            loss_fn=None,
            **imputer_kwargs
        )
        imputer.freeze()
        return imputer
    
    # 否则从exp_dir查找checkpoint
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={},
        loss_fn=None,
        **imputer_kwargs
    )

    model_path = None
    if exp_dir and os.path.exists(exp_dir):
        for file in os.listdir(exp_dir):
            if file.endswith(".ckpt"):
                model_path = os.path.join(exp_dir, file)
                break
    
    if model_path is None:
        raise ValueError(f"Model not found. 请提供 --checkpoint-path 或确保 exp_dir 中存在 .ckpt 文件")

    imputer.load_model(model_path)
    imputer.freeze()
    return imputer


def update_test_eval_mask(dm, dataset, p_fault, p_noise, seed=None):
    if seed is None:
        seed = np.random.randint(1e9)
    random = np.random.default_rng(seed)
    dataset.set_eval_mask(
        sample_mask(dataset.shape, p=p_fault, p_noise=p_noise,
                    min_seq=12, max_seq=36, rng=random)
    )
    dm.torch_dataset.set_mask(dataset.training_mask)
    dm.torch_dataset.update_exogenous('eval_mask', dataset.eval_mask)


def save_imputed_results_lane(y_hat, dataset, dm, output_path, 
                              train_indices=None, val_indices=None, test_indices=None,
                              mask_data_path=None):
    """
    将填充结果保存为与输入文件相同格式的CSV文件（用于lane数据集）
    
    Args:
        y_hat: 预测结果，形状为 [batch, window, nodes, features] 或 [window, nodes, features]
        dataset: LaneTrafficDataset 实例
        dm: SpatioTemporalDataModule 实例
        output_path: 输出文件路径
        train_indices: 训练集的索引
        val_indices: 验证集的索引
        test_indices: 测试集的索引
        mask_data_path: 掩码文件路径（可选，用于提取完整的时间和节点列表）
    """
    # 获取数据集的元信息
    timestamps = dataset.timestamps
    lane_ids = dataset.lane_ids
    feature_cols = dataset.feature_cols
    time_col = dataset.time_col
    lane_id_col = dataset.lane_id_col
    mask_time_col = dataset.mask_time_col
    mask_lane_col = dataset.mask_lane_col
    
    # 如果提供了mask文件路径，尝试从mask文件中提取完整的时间和节点列表
    if mask_data_path is not None and os.path.exists(mask_data_path):
        print(f"从mask文件中提取完整的时间和节点列表: {mask_data_path}")
        try:
            mask_df = pd.read_csv(mask_data_path)
            if mask_time_col in mask_df.columns and mask_lane_col in mask_df.columns:
                # 从mask文件中提取所有唯一的时间和节点
                mask_timestamps = np.sort(mask_df[mask_time_col].unique())
                mask_lane_ids = np.sort(mask_df[mask_lane_col].unique())
                
                # 合并dataset和mask中的时间和节点（取并集）
                all_timestamps = np.sort(np.unique(np.concatenate([timestamps, mask_timestamps])))
                all_lane_ids = np.sort(np.unique(np.concatenate([lane_ids, mask_lane_ids])))
                
                print(f"   dataset中的时间数: {len(timestamps)}, 节点数: {len(lane_ids)}")
                print(f"   mask文件中的时间数: {len(mask_timestamps)}, 节点数: {len(mask_lane_ids)}")
                print(f"   合并后的时间数: {len(all_timestamps)}, 节点数: {len(all_lane_ids)}")
                
                # 使用完整列表
                timestamps = all_timestamps
                lane_ids = all_lane_ids
            else:
                print(f"⚠️ 警告: mask文件缺少必需列 {mask_time_col} 或 {mask_lane_col}，使用dataset中的时间和节点列表")
        except Exception as e:
            print(f"⚠️ 警告: 无法从mask文件提取完整列表: {e}，使用dataset中的时间和节点列表")
    elif hasattr(dataset, 'mask_data_paths') and dataset.mask_data_paths:
        # 尝试从dataset的mask_data_paths中加载
        for mask_path in dataset.mask_data_paths:
            if mask_path is not None and os.path.exists(mask_path):
                print(f"从dataset的mask文件中提取完整的时间和节点列表: {mask_path}")
                try:
                    mask_df = pd.read_csv(mask_path)
                    if mask_time_col in mask_df.columns and mask_lane_col in mask_df.columns:
                        mask_timestamps = np.sort(mask_df[mask_time_col].unique())
                        mask_lane_ids = np.sort(mask_df[mask_lane_col].unique())
                        
                        all_timestamps = np.sort(np.unique(np.concatenate([timestamps, mask_timestamps])))
                        all_lane_ids = np.sort(np.unique(np.concatenate([lane_ids, mask_lane_ids])))
                        
                        print(f"   dataset中的时间数: {len(timestamps)}, 节点数: {len(lane_ids)}")
                        print(f"   mask文件中的时间数: {len(mask_timestamps)}, 节点数: {len(mask_lane_ids)}")
                        print(f"   合并后的时间数: {len(all_timestamps)}, 节点数: {len(all_lane_ids)}")
                        
                        timestamps = all_timestamps
                        lane_ids = all_lane_ids
                        break
                except Exception as e:
                    print(f"⚠️ 警告: 无法从mask文件 {mask_path} 提取完整列表: {e}")
                    continue
    
    # 处理y_hat的形状
    original_shape = y_hat.shape
    print(f"原始y_hat形状: {original_shape}")
    
    # 如果是4维 [batch, window, nodes, features]，需要展平
    if len(y_hat.shape) == 4:
        batch_size, window_size, n_nodes, n_features = y_hat.shape
        # 展平批次和窗口维度
        y_hat = y_hat.reshape(-1, n_nodes, n_features)
        print(f"展平后形状: {y_hat.shape}")
    
    # 确保是3维 [time, nodes, features]
    if len(y_hat.shape) != 3:
        raise ValueError(f"y_hat应该是3维或4维，但得到: {original_shape}")
    
    n_time_steps, n_nodes, n_features = y_hat.shape
    
    # 反标准化
    scaler = dm.scalers.get('data')
    if scaler is not None:
        print("执行反标准化...")
        # scaler期望 Tensor 输入，需要先转换
        y_hat_tensor = torch.from_numpy(y_hat).float()
        y_hat_inv = scaler.inverse_transform(y_hat_tensor)
        # 转换回 numpy
        if isinstance(y_hat_inv, torch.Tensor):
            y_hat = y_hat_inv.cpu().numpy()
        else:
            y_hat = y_hat_inv
    
    # 获取测试集的索引
    if test_indices is None:
        # 如果test_indices为None，使用默认的splitter参数
        splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)
        train_idx, val_idx, test_idx = splitter.split(dataset)
        test_indices = test_idx
    
    # 由于使用了窗口化，y_hat可能包含重叠的窗口
    # 我们需要从torch_dataset获取实际的索引映射
    # 简化处理：假设y_hat对应测试集的所有窗口
    # 对于窗口化的数据，我们取每个窗口的最后一个时间步（或中间时间步）
    window = dm.torch_dataset.window
    
    # 构建完整的时间序列数据
    # 由于timestamps和lane_ids可能已经扩展（从mask文件），需要扩展数据矩阵
    original_timestamps = dataset.timestamps
    original_lane_ids = dataset.lane_ids
    
    # 创建原始索引映射
    original_time_to_idx = {t: idx for idx, t in enumerate(original_timestamps)}
    original_lane_to_idx = {lid: idx for idx, lid in enumerate(original_lane_ids)}
    
    # 创建新索引映射
    new_time_to_idx = {t: idx for idx, t in enumerate(timestamps)}
    new_lane_to_idx = {lid: idx for idx, lid in enumerate(lane_ids)}
    
    # 如果时间和节点列表已扩展，需要扩展数据矩阵
    if len(timestamps) > len(original_timestamps) or len(lane_ids) > len(original_lane_ids):
        print(f"扩展数据矩阵: 从 {dataset.data.shape} 到 ({len(timestamps)}, {len(lane_ids)}, {len(feature_cols)})")
        # 创建新的数据矩阵
        n_times = len(timestamps)
        n_lanes = len(lane_ids)
        n_features = len(feature_cols)
        
        full_data = np.full((n_times, n_lanes, n_features), np.nan)
        training_mask = np.zeros((n_times, n_lanes, n_features), dtype=bool)
        
        # 从原始数据矩阵复制数据
        for orig_t_idx, orig_t in enumerate(original_timestamps):
            new_t_idx = new_time_to_idx.get(orig_t)
            if new_t_idx is not None:
                for orig_l_idx, orig_l in enumerate(original_lane_ids):
                    new_l_idx = new_lane_to_idx.get(orig_l)
                    if new_l_idx is not None:
                        full_data[new_t_idx, new_l_idx, :] = dataset.data[orig_t_idx, orig_l_idx, :]
                        training_mask[new_t_idx, new_l_idx, :] = dataset.training_mask[orig_t_idx, orig_l_idx, :]
    else:
        # 使用原始数据矩阵
        full_data = dataset.data.copy()  # [time, nodes, features]
        training_mask = dataset.training_mask.copy()  # [time, nodes, features]
    
    # 识别受路网限制的特征列（这些列如果原始值是-1，应该保持为-1）
    graph_constrained_features = ['crossing_ratio', 'direct_ratio', 'near_ratio']
    graph_constrained_indices = {}
    for feat_name in graph_constrained_features:
        if feat_name in feature_cols:
            feat_idx = feature_cols.index(feat_name)
            # 确定连接类型
            if feat_name == 'crossing_ratio':
                conn_type = 'crossing'
            elif feat_name == 'direct_ratio':
                conn_type = 'direct'
            elif feat_name == 'near_ratio':
                conn_type = 'near'
            else:
                conn_type = None
            graph_constrained_indices[feat_idx] = conn_type
    
    # 从原始输入数据中获取-1的位置（从dynamic_df读取）
    # 创建掩码：标记哪些位置应该是-1
    minus_one_mask = np.zeros_like(full_data, dtype=bool)
    if len(graph_constrained_indices) > 0 and hasattr(dataset, 'dynamic_df'):
        # 从原始CSV数据中读取-1的位置
        # 使用新的索引映射（可能已扩展）
        time_to_idx = new_time_to_idx if len(timestamps) > len(original_timestamps) else {t: idx for idx, t in enumerate(timestamps)}
        lane_id_to_idx = new_lane_to_idx if len(lane_ids) > len(original_lane_ids) else {lid: idx for idx, lid in enumerate(lane_ids)}
        
        for _, row in dataset.dynamic_df.iterrows():
            time_idx = time_to_idx.get(row[time_col])
            lane_idx = lane_id_to_idx.get(row[lane_id_col])
            
            if time_idx is not None and lane_idx is not None:
                for feat_idx, conn_type in graph_constrained_indices.items():
                    if feat_idx < len(feature_cols):
                        feat_name = feature_cols[feat_idx]
                        if feat_name in row:
                            val = row[feat_name]
                            # 如果原始数据中是-1，标记为应该保持-1
                            if pd.notna(val) and val == -1.0:
                                minus_one_mask[time_idx, lane_idx, feat_idx] = True
    
    # 获取窗口和步长信息
    window = dm.torch_dataset.window
    stride = dm.torch_dataset.stride
    
    # 将窗口化的预测结果映射回完整时间序列
    # y_hat 的形状是 [num_windows, window, nodes, features] 或 [num_windows * window, nodes, features]
    # 需要根据窗口的起始位置映射回原始时间序列
    
    # 获取所有数据集的索引（用于确定窗口的起始位置）
    all_indices = []
    if train_indices is not None:
        all_indices.extend(train_indices)
    if val_indices is not None:
        all_indices.extend(val_indices)
    if test_indices is not None:
        all_indices.extend(test_indices)
    
    # 转换索引为整数（如果是时间戳）
    def convert_to_int_indices(indices):
        if indices is None or len(indices) == 0:
            return []
        first_idx = indices[0]
        if isinstance(first_idx, (str, pd.Timestamp)) or hasattr(first_idx, 'strftime'):
            time_to_idx = {t: idx for idx, t in enumerate(timestamps)}
            int_indices = [time_to_idx.get(t, -1) for t in indices]
            return [idx for idx in int_indices if idx >= 0]
        else:
            return [int(idx) for idx in indices]
    
    train_int = convert_to_int_indices(train_indices)
    val_int = convert_to_int_indices(val_indices)
    test_int = convert_to_int_indices(test_indices)
    all_int = sorted(set(train_int + val_int + test_int))
    
    # 计算每个数据集对应的窗口范围
    # 由于窗口化，我们需要知道每个窗口对应的时间步
    # 简化处理：假设窗口按顺序排列，每个窗口的最后一个时间步对应预测值
    
    # 计算窗口数量
    num_windows = n_time_steps // window if n_time_steps >= window else 1
    
    # 将y_hat重塑为 [num_windows, window, nodes, features]
    if len(y_hat.shape) == 3:
        # 如果是 [time, nodes, features]，需要重塑
        if n_time_steps % window == 0:
            y_hat_reshaped = y_hat.reshape(num_windows, window, n_nodes, n_features)
        else:
            # 如果不能整除，填充或截断
            target_size = num_windows * window
            if n_time_steps < target_size:
                # 填充
                padding = np.zeros((target_size - n_time_steps, n_nodes, n_features))
                y_hat_padded = np.concatenate([y_hat, padding], axis=0)
                y_hat_reshaped = y_hat_padded.reshape(num_windows, window, n_nodes, n_features)
            else:
                # 截断
                y_hat_reshaped = y_hat[:target_size].reshape(num_windows, window, n_nodes, n_features)
    else:
        y_hat_reshaped = y_hat
    
    # 映射窗口预测到完整时间序列
    # 注意：y_hat只包含原始时间范围内的预测结果，需要映射到原始时间范围
    # 对于新增的时间（在mask文件中但不在原始数据中），这些位置保持为NaN
    original_n_times = len(original_timestamps)
    
    # 映射窗口预测到原始时间序列（只映射到原始时间范围内的数据）
    window_idx = 0
    for start_time in range(0, original_n_times - window + 1, stride):
        if window_idx >= num_windows:
            break
        
        # 获取该窗口的所有时间步预测 [window, nodes, features]
        window_preds = y_hat_reshaped[window_idx, :, :, :]
        # 将该窗口的所有时间步映射回原始时间序列
        for w in range(window):
            orig_time_idx = start_time + w
            if orig_time_idx < original_n_times:
                # 获取原始时间戳
                orig_timestamp = original_timestamps[orig_time_idx]
                # 找到在新时间列表中的索引
                new_time_idx = new_time_to_idx.get(orig_timestamp)
                
                if new_time_idx is not None:
                    # window_pred的形状是[原始节点数, features]，需要只映射到原始节点范围
                    window_pred = window_preds[w, :, :]  # [原始节点数, features]
                    
                    # 只映射到原始节点范围内的数据
                    for orig_l_idx, orig_lane_id in enumerate(original_lane_ids):
                        new_l_idx = new_lane_to_idx.get(orig_lane_id)
                        if new_l_idx is not None:
                            # 对于缺失值位置，使用预测值；对于已知值位置，保留原始值
                            mask_missing = ~training_mask[new_time_idx, new_l_idx, :]  # 缺失值的位置 [features]
                            window_pred_lane = window_pred[orig_l_idx, :]  # [features]
                            full_data[new_time_idx, new_l_idx, :] = np.where(
                                mask_missing, window_pred_lane, full_data[new_time_idx, new_l_idx, :]
                            )
                            
                            # 对于受路网限制的特征，如果原始数据是-1，则保持为-1
                            if len(graph_constrained_indices) > 0:
                                for feat_idx in graph_constrained_indices:
                                    if feat_idx < full_data.shape[-1]:
                                        # 如果原始数据中该位置是-1，则保持为-1
                                        if minus_one_mask[new_time_idx, new_l_idx, feat_idx]:
                                            full_data[new_time_idx, new_l_idx, feat_idx] = -1.0
        
        window_idx += 1
    
    # 如果还有剩余的预测值，处理最后一个窗口
    if window_idx < num_windows and original_n_times > 0:
        # 计算最后一个窗口的起始位置（基于原始时间范围）
        last_start = original_n_times - window
        if last_start >= 0:
            window_preds = y_hat_reshaped[window_idx, :, :, :]
            for w in range(window):
                orig_time_idx = last_start + w
                if orig_time_idx < original_n_times:
                    orig_timestamp = original_timestamps[orig_time_idx]
                    new_time_idx = new_time_to_idx.get(orig_timestamp)
                    
                    if new_time_idx is not None:
                        # window_pred的形状是[原始节点数, features]，需要只映射到原始节点范围
                        window_pred = window_preds[w, :, :]  # [原始节点数, features]
                        
                        # 只映射到原始节点范围内的数据
                        for orig_l_idx, orig_lane_id in enumerate(original_lane_ids):
                            new_l_idx = new_lane_to_idx.get(orig_lane_id)
                            if new_l_idx is not None:
                                # 对于缺失值位置，使用预测值；对于已知值位置，保留原始值
                                mask_missing = ~training_mask[new_time_idx, new_l_idx, :]  # 缺失值的位置 [features]
                                window_pred_lane = window_pred[orig_l_idx, :]  # [features]
                                full_data[new_time_idx, new_l_idx, :] = np.where(
                                    mask_missing, window_pred_lane, full_data[new_time_idx, new_l_idx, :]
                                )
                                
                                # 对于受路网限制的特征，如果原始数据是-1，则保持为-1
                                if len(graph_constrained_indices) > 0:
                                    for feat_idx in graph_constrained_indices:
                                        if feat_idx < full_data.shape[-1]:
                                            # 如果原始数据中该位置是-1，则保持为-1
                                            if minus_one_mask[new_time_idx, new_l_idx, feat_idx]:
                                                full_data[new_time_idx, new_l_idx, feat_idx] = -1.0
    
    # 构建DataFrame
    # 注意：先遍历lane_id，再遍历timestamp，这样构建的数据已经是按lane_id优先的顺序
    result_rows = []
    for n_idx, lane_id in enumerate(lane_ids):
        for t_idx, timestamp in enumerate(timestamps):
            row = {
                lane_id_col: lane_id,  # lane_id放在第一列
                time_col: timestamp     # time_col放在第二列
            }
            # 添加所有特征列
            for f_idx, feature_name in enumerate(feature_cols):
                if f_idx < full_data.shape[-1]:
                    row[feature_name] = full_data[t_idx, n_idx, f_idx]
                else:
                    row[feature_name] = np.nan
            result_rows.append(row)
    
    # 创建DataFrame并保存
    result_df = pd.DataFrame(result_rows)
    
    # 按lane_id优先排序，然后按时间排序（确保顺序：lane_id=0的所有时间，lane_id=1的所有时间...）
    result_df = result_df.sort_values([lane_id_col, time_col])
    
    # 调整列顺序：确保lane_id_col是第一列，time_col是第二列，然后是特征列
    column_order = [lane_id_col, time_col] + feature_cols
    # 只保留实际存在的列
    column_order = [col for col in column_order if col in result_df.columns]
    result_df = result_df[column_order]
    
    # 格式化数值列：整数保持整数，1位小数保持1位，2位及以上四舍五入到2位
    def format_number(x):
        """格式化数值：保持整数和1位小数的原始格式，2位及以上四舍五入到2位"""
        if pd.isna(x):
            return x
        # 检查是否为整数（考虑浮点误差）
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        # 检查是否为1位小数
        rounded_1 = round(x, 1)
        if abs(x - rounded_1) < 1e-10:
            return rounded_1
        # 否则四舍五入到2位小数
        return round(x, 2)
    
    numeric_cols = [col for col in result_df.columns 
                   if col not in [time_col, lane_id_col]]
    for col in numeric_cols:
        result_df[col] = result_df[col].apply(format_number)
    
    # 保存为CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"✅ 填充结果已保存到: {output_path}")
    print(f"   共 {len(result_df)} 条记录，{len(feature_cols)} 个特征列")
    print(f"   时间范围: {result_df[time_col].min()} 到 {result_df[time_col].max()}")
    
    return result_df


def run_experiment(args):
    # Set configuration
    args = copy.deepcopy(args)
    tsl.logger.disabled = True

    # script flags
    is_spin = args.model_name in ['spin', 'spin_h']

    ########################################
    # load config                          #
    ########################################

    exp_dir = None
    exp_config = {}
    
    # 如果提供了checkpoint_path，尝试从checkpoint所在目录加载config
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as fp:
                exp_config = yaml.load(fp, Loader=yaml.FullLoader)
            exp_dir = checkpoint_dir
            print(f"从checkpoint目录加载配置: {config_path}")
        else:
            print(f"警告: checkpoint目录中未找到config.yaml，将使用命令行参数")
    
    # 如果没有从checkpoint加载到config，尝试从exp_dir加载
    if not exp_config and args.exp_name is not None:
        if args.root is None:
            root = tsl.config.log_dir
        else:
            root = os.path.join(tsl.config.curr_dir, args.root)
        exp_dir = os.path.join(root, args.dataset_name,
                               args.model_name, args.exp_name)
        config_path = os.path.join(exp_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as fp:
                exp_config = yaml.load(fp, Loader=yaml.FullLoader)
            print(f"从实验目录加载配置: {config_path}")
    
    # 如果仍然没有config，使用命令行参数作为默认值
    if not exp_config:
        print("警告: 未找到配置文件，使用命令行参数作为默认值")
        exp_config = vars(args)
        if args.model_name is None:
            raise ValueError("必须提供 --model-name 参数")
        exp_config['model_name'] = args.model_name
        if args.dataset_name is None:
            raise ValueError("必须提供 --dataset-name 参数")
        exp_config['dataset_name'] = args.dataset_name
        # 设置默认的window和stride（如果未提供）
        if 'window' not in exp_config:
            exp_config['window'] = 10
        if 'stride' not in exp_config:
            exp_config['stride'] = 1

    ########################################
    # load dataset                         #
    ########################################

    # 解析特征列
    feature_cols = None
    if args.feature_cols:
        feature_cols = [col.strip() for col in args.feature_cols.split(',')]
    
    dataset_name = exp_config.get('dataset_name', args.dataset_name)
    dataset = get_dataset(
        dataset_name,
        data_path=args.data_path,
        static_data_path=args.static_data_path,
        mask_data_path=args.mask_data_path,
        feature_cols=feature_cols
    )

    ########################################
    # load data module                     #
    ########################################

    # time embedding
    u_size = 1  # 默认值
    if is_spin or args.model_name == 'transformer':
        # lane数据集使用空的时间编码列表
        if dataset_name == 'lane':
            time_emb = dataset.datetime_encoded([]).values
        else:
            time_emb = dataset.datetime_encoded(['day', 'week']).values
        # 获取时间编码的实际维度
        u_size = time_emb.shape[-1] if len(time_emb.shape) > 1 else 1
        exog_map = {'global_temporal_encoding': time_emb}

        input_map = {
            'u': 'temporal_encoding',
            'x': 'data'
        }
    else:
        exog_map = input_map = None

    if is_spin or args.model_name == 'grin':
        adj = dataset.get_connectivity(threshold=args.adj_threshold,
                                       include_self=False,
                                       force_symmetric=is_spin)
        # 将邻接矩阵转换为 edge_index 格式 (2, num_edges)
        from tsl.ops.connectivity import adj_to_edge_index
        edge_index, edge_weight = adj_to_edge_index(adj)
        connectivity = (edge_index, edge_weight)
    else:
        connectivity = None

    # instantiate dataset
    data, index, node_ids = dataset.numpy(return_idx=True)
    torch_dataset = ImputationDataset(data=data,
                                      index=index,
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=connectivity,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=exp_config.get('window', 10),
                                      stride=exp_config.get('stride', 1))

    # get train/val/test indices
    splitter = dataset.get_splitter(args.val_len, args.test_len)

    scalers = {'data': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset,
                                  scalers=scalers,
                                  splitter=splitter,
                                  batch_size=args.batch_size)
    dm.setup()

    ########################################
    # load model                           #
    ########################################

    imputer = load_model(exp_dir, exp_config, dm, 
                        checkpoint_path=args.checkpoint_path,
                        u_size=u_size)

    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        devices=1 if torch.cuda.is_available() else None)

    ########################################
    # inference                            #
    ########################################

    seeds = ensure_list(args.test_mask_seed) if args.test_mask_seed is not None else [None]
    mae = []

    for seed in seeds:
        # Change evaluation mask (仅对非lane数据集，lane数据集使用自己的mask)
        if dataset_name != 'lane':
            update_test_eval_mask(dm, dataset, args.p_fault, args.p_noise, seed)

        output_list = trainer.predict(imputer, dataloaders=dm.test_dataloader())
        
        # 将字典列表合并为单个字典，每个键包含所有批次的拼接结果
        # output_list 是一个字典列表，每个字典包含 'y_hat', 'y', 'mask'
        y_hat_list = []
        y_list = []
        mask_list = []
        
        for batch_output in output_list:
            y_hat_list.append(batch_output['y_hat'].detach().cpu())
            y_list.append(batch_output['y'].detach().cpu())
            mask_list.append(batch_output['mask'].detach().cpu())
        
        # 拼接所有批次
        y_hat = torch.cat(y_hat_list, dim=0).numpy()
        y_true = torch.cat(y_list, dim=0).numpy()
        mask = torch.cat(mask_list, dim=0).numpy()
        
        # 只在最后一个维度大小为1时才squeeze
        if y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)
        if y_true.shape[-1] == 1:
            y_true = y_true.squeeze(-1)
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
        mae.append(check_mae)
        seed_str = f'SEED {seed}' if seed is not None else 'NO SEED'
        print(f'{seed_str} - Test MAE: {check_mae:.3f}')
        
        # 保存填充结果（仅对lane数据集，且只保存第一个seed的结果）
        if dataset_name == 'lane' and args.output_path is not None and seed == seeds[0]:
            print(f"\n保存填充结果...")
            # 对所有数据进行推理（不仅仅是测试集）
            print("对所有数据进行推理...")
            all_output_list = []
            
            # 训练集
            train_dl = dm.train_dataloader()
            if train_dl is not None:
                train_output = trainer.predict(imputer, dataloaders=train_dl)
                if train_output is not None:
                    all_output_list.extend(train_output)
            
            # 验证集
            val_dl = dm.val_dataloader()
            if val_dl is not None:
                val_output = trainer.predict(imputer, dataloaders=val_dl)
                if val_output is not None:
                    all_output_list.extend(val_output)
            
            # 测试集
            test_dl = dm.test_dataloader()
            if test_dl is not None:
                test_output = trainer.predict(imputer, dataloaders=test_dl)
                if test_output is not None:
                    all_output_list.extend(test_output)
            
            # 合并所有批次的预测结果
            all_y_hat_list = []
            for batch_output in all_output_list:
                all_y_hat_list.append(batch_output['y_hat'].detach().cpu())
            
            # 拼接所有批次
            all_y_hat = torch.cat(all_y_hat_list, dim=0).numpy()
            
            # 只在最后一个维度大小为1时才squeeze
            if all_y_hat.shape[-1] == 1:
                all_y_hat = all_y_hat.squeeze(-1)
            
            # 获取所有索引
            train_idx, val_idx, test_idx = splitter.split(dataset)
            save_imputed_results_lane(all_y_hat, dataset, dm, args.output_path, 
                                     train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
                                     mask_data_path=args.mask_data_path)

    if len(mae) > 1:
        print(f'MAE over {len(seeds)} runs: {np.mean(mae):.3f}±{np.std(mae):.3f}')
    else:
        print(f'Test MAE: {mae[0]:.3f}')


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
