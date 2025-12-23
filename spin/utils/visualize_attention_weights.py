"""
SPIN模型注意力权重可视化工具

使用示例:
    from spin.utils.visualize_attention_weights import AttentionVisualizer
    
    visualizer = AttentionVisualizer(model)
    spatial_weights, temporal_weights = visualizer.extract_weights(x, mask, edge_index)
    visualizer.plot_temporal_attention(temporal_weights, layer_idx=0, save_path='temp_attn.png')
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import copy


class AttentionVisualizer:
    """注意力权重可视化工具类"""
    
    def __init__(self, model):
        self.model = model
        self.original_methods = {}
        self.extracted_weights = {}
        
    def _patch_additive_attention(self):
        """临时修改AdditiveAttention的message方法以提取权重"""
        from spin.layers.additive_attention import AdditiveAttention
        
        # 保存原始方法
        if 'AdditiveAttention.message' not in self.original_methods:
            self.original_methods['AdditiveAttention.message'] = AdditiveAttention.message
        
        def message_with_extraction(self, msg_j, msg_i, index, size_i, mask_j=None):
            """带权重提取的message方法"""
            msg = self.msg_nn(msg_j + msg_i)
            gate = self.msg_gate(msg)
            alpha = self.normalize_weights(gate, index, size_i, mask_j)
            
            # 保存权重
            if hasattr(self, '_save_attention_weights') and self._save_attention_weights:
                if not hasattr(self, '_attention_weights_buffer'):
                    self._attention_weights_buffer = []
                self._attention_weights_buffer.append({
                    'weights': alpha.detach().cpu().clone(),
                    'index': index.detach().cpu().clone() if isinstance(index, torch.Tensor) else index,
                    'size_i': size_i,
                    'msg_shape': msg.shape
                })
            
            # 应用dropout（与原始代码保持一致）
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            out = alpha * msg
            return out
        
        # 临时替换方法
        AdditiveAttention.message = message_with_extraction
        
    def _restore_additive_attention(self):
        """恢复原始的message方法"""
        from spin.layers.additive_attention import AdditiveAttention
        if 'AdditiveAttention.message' in self.original_methods:
            AdditiveAttention.message = self.original_methods['AdditiveAttention.message']
    
    def extract_weights(self, x: torch.Tensor, mask: torch.Tensor, 
                       edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,
                       u: Optional[torch.Tensor] = None):
        """
        提取模型的注意力权重
        
        Args:
            x: 输入数据 [B, T, N, C]
            mask: mask [B, T, N, C]
            edge_index: 边索引 [2, E]
            edge_weight: 边权重（可选）
            u: 时间特征（可选）
        
        Returns:
            spatial_weights: 空间注意力权重字典 {layer_idx: weights_data}
            temporal_weights: 时间注意力权重字典 {layer_idx: weights_data}
        """
        self.model.eval()
        
        # 启用权重提取
        self._patch_additive_attention()
        
        # 为所有注意力层启用权重保存
        layer_spatial = {}
        layer_temporal = {}
        
        for layer_idx, encoder_layer in enumerate(self.model.encoder):
            # 空间注意力
            if hasattr(encoder_layer, 'cross_attention'):
                encoder_layer.cross_attention._save_attention_weights = True
                encoder_layer.cross_attention._attention_weights_buffer = []
                layer_spatial[layer_idx] = encoder_layer.cross_attention
            
            # 时间注意力
            if hasattr(encoder_layer, 'self_attention') and encoder_layer.self_attention is not None:
                encoder_layer.self_attention._save_attention_weights = True
                encoder_layer.self_attention._attention_weights_buffer = []
                layer_temporal[layer_idx] = encoder_layer.self_attention
        
        # 执行前向传播
        with torch.no_grad():
            output = self.model(x, u=u, mask=mask, edge_index=edge_index, edge_weight=edge_weight)
        
        # 收集权重
        spatial_weights = {}
        temporal_weights = {}
        
        for layer_idx, attention_layer in layer_spatial.items():
            if hasattr(attention_layer, '_attention_weights_buffer'):
                spatial_weights[layer_idx] = attention_layer._attention_weights_buffer.copy()
                attention_layer._save_attention_weights = False
                attention_layer._attention_weights_buffer = []
        
        for layer_idx, attention_layer in layer_temporal.items():
            if hasattr(attention_layer, '_attention_weights_buffer'):
                temporal_weights[layer_idx] = attention_layer._attention_weights_buffer.copy()
                attention_layer._save_attention_weights = False
                attention_layer._attention_weights_buffer = []
        
        # 恢复原始方法
        self._restore_additive_attention()
        
        return spatial_weights, temporal_weights
    
    def reconstruct_temporal_matrix(self, weights_data: List[Dict], 
                                   edge_index: torch.Tensor,
                                   n_time_steps: int,
                                   batch_idx: int = 0) -> np.ndarray:
        """
        从提取的权重重构时间注意力矩阵
        
        Args:
            weights_data: 从extract_weights返回的权重数据列表
            edge_index: 时间步之间的边索引 [2, E]
            n_time_steps: 时间步数
            batch_idx: 批次索引
        
        Returns:
            attention_matrix: [n_time_steps, n_time_steps] 的注意力权重矩阵
        """
        if not weights_data:
            return np.zeros((n_time_steps, n_time_steps))
        
        # 使用最后一个（通常是完整前向传播的结果）
        weight_item = weights_data[-1]
        weights = weight_item['weights'].numpy()  # [B, T, E, 1] or [E, 1]
        indices = weight_item['index'].numpy() if isinstance(weight_item['index'], torch.Tensor) else weight_item['index']
        
        edge_index_np = edge_index.detach().cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        
        # 处理批次维度
        if weights.ndim == 4:  # [B, T, E, 1]
            weights = weights[batch_idx, :, :, 0]  # [T, E]
            # 对每个时间步，聚合边权重
            attention_matrix = np.zeros((n_time_steps, n_time_steps))
            for t in range(n_time_steps):
                for e_idx in range(min(weights.shape[1], edge_index_np.shape[1])):
                    src, tgt = edge_index_np[0, e_idx], edge_index_np[1, e_idx]
                    if src < n_time_steps and tgt < n_time_steps:
                        attention_matrix[tgt, src] = weights[t, e_idx]
        else:  # [E, 1] or [E]
            attention_matrix = np.zeros((n_time_steps, n_time_steps))
            weights_flat = weights.flatten() if weights.ndim > 1 else weights
            for e_idx in range(min(len(weights_flat), edge_index_np.shape[1])):
                src, tgt = edge_index_np[0, e_idx], edge_index_np[1, e_idx]
                if src < n_time_steps and tgt < n_time_steps:
                    attention_matrix[tgt, src] = weights_flat[e_idx]
        
        return attention_matrix
    
    def plot_temporal_attention(self, temporal_weights: Dict, layer_idx: int,
                               edge_index: torch.Tensor, n_time_steps: int,
                               batch_idx: int = 0, node_idx: int = 0,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 8),
                               title_suffix: str = ""):
        """
        绘制时间注意力权重热力图
        
        Args:
            temporal_weights: 从extract_weights返回的时间权重字典
            layer_idx: 层索引
            edge_index: 时间步之间的边索引（用于重构矩阵）
            n_time_steps: 时间步数
            batch_idx: 批次索引
            node_idx: 节点索引（用于标题）
            save_path: 保存路径
            figsize: 图形大小
            title_suffix: 标题后缀
        """
        if layer_idx not in temporal_weights:
            print(f"Warning: No temporal weights found for layer {layer_idx}")
            return
        
        # 重构注意力矩阵
        attn_matrix = self.reconstruct_temporal_matrix(
            temporal_weights[layer_idx], edge_index, n_time_steps, batch_idx
        )
        
        # 绘制热力图
        plt.figure(figsize=figsize)
        sns.heatmap(attn_matrix,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Attention Weight'},
                   square=True,
                   linewidths=0.5)
        plt.title(f'Temporal Attention Weights - Layer {layer_idx}, Node {node_idx}' + 
                 (f' ({title_suffix})' if title_suffix else ''))
        plt.xlabel('Source Time Step')
        plt.ylabel('Target Time Step')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved temporal attention visualization to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_spatial_attention_stats(self, spatial_weights: Dict, layer_idx: int,
                                    edge_index: torch.Tensor,
                                    batch_idx: int = 0, time_step: int = 0,
                                    top_k: int = 20, save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (12, 6)):
        """
        绘制空间注意力权重的统计信息（Top-K边）
        
        Args:
            spatial_weights: 从extract_weights返回的空间权重字典
            layer_idx: 层索引
            edge_index: 空间边索引 [2, E]
            batch_idx: 批次索引
            time_step: 时间步索引
            top_k: 显示前K个最高权重的边
            save_path: 保存路径
            figsize: 图形大小
        """
        if layer_idx not in spatial_weights:
            print(f"Warning: No spatial weights found for layer {layer_idx}")
            return
        
        weights_data = spatial_weights[layer_idx][-1]  # 使用最后一个
        weights = weights_data['weights'].numpy()
        edge_index_np = edge_index.detach().cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        
        # 处理批次和时间维度
        if weights.ndim == 4:  # [B, T, E, 1]
            weights_flat = weights[batch_idx, time_step, :, 0]  # [E]
        elif weights.ndim == 3:  # [B, E, 1]
            weights_flat = weights[batch_idx, :, 0]  # [E]
        else:  # [E, 1] or [E]
            weights_flat = weights.flatten() if weights.ndim > 1 else weights
        
        # 获取Top-K边
        n_edges = min(len(weights_flat), edge_index_np.shape[1], top_k)
        top_indices = np.argsort(weights_flat)[-n_edges:][::-1]
        
        # 准备数据
        edges = []
        edge_weights = []
        for idx in top_indices:
            src, tgt = edge_index_np[0, idx], edge_index_np[1, idx]
            edges.append(f"({src}→{tgt})")
            edge_weights.append(weights_flat[idx])
        
        # 绘制柱状图
        plt.figure(figsize=figsize)
        plt.barh(range(len(edges)), edge_weights, color='steelblue')
        plt.yticks(range(len(edges)), edges)
        plt.xlabel('Attention Weight')
        plt.title(f'Top-{n_edges} Spatial Attention Weights - Layer {layer_idx}, Time Step {time_step}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved spatial attention stats to {save_path}")
        else:
            plt.show()
        plt.close()


def create_temporal_edge_index(n_time_steps: int, 
                               max_distance: Optional[int] = None,
                               causal: bool = False) -> torch.Tensor:
    """
    创建时间步之间的边索引（用于重构时间注意力矩阵）
    
    Args:
        n_time_steps: 时间步数
        max_distance: 最大时间距离（None表示全连接）
        causal: 是否只连接过去的时间步
    
    Returns:
        edge_index: [2, E] 的边索引，edge_index[0]是源时间步，edge_index[1]是目标时间步
    """
    edges = []
    for tgt in range(n_time_steps):
        for src in range(n_time_steps):
            if causal and src > tgt:
                continue
            if max_distance is not None:
                if abs(src - tgt) > max_distance:
                    continue
            edges.append([src, tgt])
    
    return torch.tensor(edges, dtype=torch.long).T


# 使用示例函数
def example_usage():
    """使用示例"""
    # 假设你有模型和数据
    # model = SPINModel(...)
    # x = torch.randn(1, 24, 10, 1)  # [B, T, N, C]
    # mask = torch.ones(1, 24, 10, 1)
    # edge_index = torch.randint(0, 10, (2, 20))  # 空间边
    
    # 创建可视化工具
    # visualizer = AttentionVisualizer(model)
    
    # 提取权重
    # spatial_weights, temporal_weights = visualizer.extract_weights(x, mask, edge_index)
    
    # 创建时间边索引（需要根据实际的时间连接方式）
    # temporal_edge_index = create_temporal_edge_index(n_time_steps=24, max_distance=5)
    
    # 可视化时间注意力
    # visualizer.plot_temporal_attention(
    #     temporal_weights, layer_idx=0,
    #     edge_index=temporal_edge_index, n_time_steps=24,
    #     save_path='temporal_attention_layer0.png'
    # )
    
    # 可视化空间注意力统计
    # visualizer.plot_spatial_attention_stats(
    #     spatial_weights, layer_idx=0,
    #     edge_index=edge_index,
    #     save_path='spatial_attention_layer0.png'
    # )
    pass

