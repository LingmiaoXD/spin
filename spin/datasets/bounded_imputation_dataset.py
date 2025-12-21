"""
自定义的 ImputationDataset，用于过滤跨越文件边界的窗口
避免模型学习到不同文件之间的错误时间关系
"""

from typing import Optional, List, Tuple
import numpy as np
import torch
from tsl.data import ImputationDataset


def filter_cross_boundary_windows(torch_dataset: ImputationDataset,
                                  file_boundaries: List[Tuple[int, int]],
                                  window_size: int) -> ImputationDataset:
    """
    过滤掉跨越文件边界的窗口索引
    
    当使用多个文件进行训练时，如果文件在时间上不连续，直接拼接会导致
    模型学习到错误的时间关系。这个函数通过记录文件边界，确保每个窗口
    只包含来自同一个文件的数据。
    
    Args:
        torch_dataset: ImputationDataset 实例
        file_boundaries: 文件边界列表，每个元素是 (start_idx, end_idx)，
                        表示文件在时间索引中的范围（end_idx 是开区间）
        window_size: 窗口大小
        
    Returns:
        修改后的 ImputationDataset（已过滤 indices）
    """
    if not file_boundaries:
        print("⚠️ 警告: 未提供文件边界信息，将使用原始数据集（可能包含跨越边界的窗口）")
        return torch_dataset
    
    def is_window_valid(window_start_idx: int, window_size: int) -> bool:
        """
        检查窗口是否有效（不跨越文件边界）
        
        Args:
            window_start_idx: 窗口开始的时间索引
            window_size: 窗口大小
            
        Returns:
            bool: True 如果窗口有效（不跨越边界），False 否则
        """
        window_end_idx = window_start_idx + window_size
        
        # 检查窗口是否完全在某个文件范围内
        for start_bound, end_bound in file_boundaries:
            # 窗口必须完全在某个文件范围内
            if window_start_idx >= start_bound and window_end_idx <= end_bound:
                return True
        
        return False
    
    # 获取原始的有效索引
    if not hasattr(torch_dataset, 'indices') or torch_dataset.indices is None:
        print("⚠️ 警告: 数据集没有 indices 属性，无法过滤窗口")
        return torch_dataset
    
    # 处理不同类型的 indices（Tensor、numpy array、list）
    indices = torch_dataset.indices
    is_tensor = torch.is_tensor(indices)
    
    if is_tensor:
        # 如果是 Tensor，转换为 numpy 数组进行处理，但记住原始类型
        original_indices = indices.cpu().numpy().copy()
    elif isinstance(indices, np.ndarray):
        # 如果是 numpy 数组，使用 copy()
        original_indices = indices.copy()
    elif isinstance(indices, (list, tuple)):
        # 如果是列表或元组，转换为列表
        original_indices = list(indices)
    else:
        # 其他情况，尝试转换为 numpy 数组
        try:
            original_indices = np.array(indices).copy()
        except:
            print("⚠️ 警告: 无法处理 indices 类型，将使用原始数据集")
            return torch_dataset
    
    valid_indices = []
    
    for idx in original_indices:
        # 假设 idx 是窗口的起始时间索引
        if isinstance(idx, (int, np.integer)):
            window_start_idx = int(idx)
        elif isinstance(idx, (list, tuple, np.ndarray)):
            # 如果 idx 是数组，取第一个元素作为起始索引
            window_start_idx = int(idx[0]) if len(idx) > 0 else 0
        else:
            # 其他情况，尝试转换
            try:
                window_start_idx = int(idx)
            except:
                # 如果无法转换，保留原始索引（向后兼容）
                valid_indices.append(idx)
                continue
        
        # 检查窗口是否有效
        if is_window_valid(window_start_idx, window_size):
            valid_indices.append(idx)
    
    # 更新 indices - 保持原始类型（Tensor 或 numpy array）
    if len(valid_indices) > 0:
        if is_tensor:
            # 如果是 Tensor，转换回 Tensor
            filtered_indices = torch.tensor(valid_indices, dtype=indices.dtype, device=indices.device)
        else:
            # 否则保持为 numpy 数组
            filtered_indices = np.array(valid_indices)
    else:
        if is_tensor:
            # 如果是 Tensor，创建空的 Tensor
            filtered_indices = torch.tensor([], dtype=indices.dtype, device=indices.device)
        else:
            # 否则创建空的 numpy 数组
            filtered_indices = np.array([], dtype=int)
    
    # 尝试直接设置 _indices（如果存在）或使用 object.__setattr__ 绕过属性设置器
    if hasattr(torch_dataset, '_indices'):
        object.__setattr__(torch_dataset, '_indices', filtered_indices)
    else:
        # 使用 object.__setattr__ 直接设置 indices，绕过属性设置器
        object.__setattr__(torch_dataset, 'indices', filtered_indices)
    
    filtered_count = len(original_indices) - len(filtered_indices)
    print(f"✅ 已过滤 {filtered_count} 个跨越文件边界的窗口")
    print(f"   原始窗口数: {len(original_indices)}, 剩余有效窗口数: {len(filtered_indices)}")
    
    return torch_dataset


class BoundedImputationDataset(ImputationDataset):
    """
    继承自 ImputationDataset，但会过滤掉跨越文件边界的窗口
    
    这是一个包装类，在初始化后自动过滤 indices。
    """
    
    def __init__(self, 
                 file_boundaries: Optional[List[Tuple[int, int]]] = None,
                 *args, **kwargs):
        """
        初始化有边界限制的数据集
        
        Args:
            file_boundaries: 文件边界列表，每个元素是 (start_idx, end_idx)，
                            表示文件在时间索引中的范围（end_idx 是开区间）
            *args, **kwargs: 传递给 ImputationDataset 的其他参数
        """
        super().__init__(*args, **kwargs)
        
        # 过滤跨越边界的窗口
        if file_boundaries:
            window_size = kwargs.get('window', getattr(self, 'window', 10))
            filter_cross_boundary_windows(self, file_boundaries, window_size)

