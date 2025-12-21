"""
自定义的 ImputationDataset，用于过滤跨越文件边界的窗口
避免模型学习到不同文件之间的错误时间关系
"""

from typing import Optional, List, Tuple
import numpy as np
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
    
    original_indices = torch_dataset.indices.copy()
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
    
    # 更新 indices
    if isinstance(valid_indices, list):
        torch_dataset.indices = np.array(valid_indices) if len(valid_indices) > 0 else np.array([], dtype=int)
    else:
        torch_dataset.indices = valid_indices
    
    filtered_count = len(original_indices) - len(torch_dataset.indices)
    print(f"✅ 已过滤 {filtered_count} 个跨越文件边界的窗口")
    print(f"   原始窗口数: {len(original_indices)}, 剩余有效窗口数: {len(torch_dataset.indices)}")
    
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

