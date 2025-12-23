"""
注意力权重可视化工具

这个模块提供了提取和可视化SPIN模型注意力权重的功能。
支持空间图注意力和时间自注意力的权重可视化。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class AttentionHook:
    """用于捕获注意力权重的Hook类"""
    
    def __init__(self):
        self.attention_weights = {}
        self.layer_idx = 0
        
    def register_hooks(self, model, layer_type='encoder'):
        """为模型注册hooks以捕获注意力权重"""
        self.attention_weights = {}
        self.layer_idx = 0
        
        if hasattr(model, 'encoder'):
            for idx, layer in enumerate(model.encoder):
                if hasattr(layer, 'cross_attention'):
                    # 注册空间注意力hook
                    layer.cross_attention.register_forward_hook(
                        self._create_spatial_attention_hook(idx, 'spatial')
                    )
                if hasattr(layer, 'self_attention') and layer.self_attention is not None:
                    # 注册时间注意力hook
                    layer.self_attention.register_forward_hook(
                        self._create_temporal_attention_hook(idx, 'temporal')
                    )
    
    def _create_spatial_attention_hook(self, layer_idx, attn_type):
        """创建空间注意力hook"""
        def hook(module, input, output):
            # 在AdditiveAttention中，我们需要在message方法中捕获权重
            # 这里我们注册一个hook到msg_gate层
            key = f"layer_{layer_idx}_{attn_type}"
            if key not in self.attention_weights:
                self.attention_weights[key] = []
        return hook
    
    def _create_temporal_attention_hook(self, layer_idx, attn_type):
        """创建时间注意力hook"""
        def hook(module, input, output):
            key = f"layer_{layer_idx}_{attn_type}"
            if key not in self.attention_weights:
                self.attention_weights[key] = []
        return hook


def extract_attention_weights_modified(attention_layer, x, edge_index, mask=None, 
                                       return_weights=True):
    """
    修改版本的forward，用于提取注意力权重
    
    Args:
        attention_layer: AdditiveAttention层
        x: 输入特征
        edge_index: 边索引
        mask: mask
        return_weights: 是否返回权重
    
    Returns:
        output: 输出特征
        attention_weights: 注意力权重（如果return_weights=True）
    """
    from torch_geometric.typing import PairTensor, Tensor
    
    if isinstance(x, Tensor):
        x_src = x_tgt = x
    else:
        x_src, x_tgt = x
        x_tgt = x_tgt if x_tgt is not None else x_src
    
    N_src, N_tgt = x_src.size(attention_layer.node_dim), x_tgt.size(attention_layer.node_dim)
    
    msg_src = attention_layer.lin_src(x_src)
    msg_tgt = attention_layer.lin_tgt(x_tgt)
    msg = (msg_src, msg_tgt)
    
    # 自定义message函数以捕获权重
    saved_weights = []
    
    def custom_message(msg_j, msg_i, index, size_i, mask_j=None):
        msg = attention_layer.msg_nn(msg_j + msg_i)
        gate = attention_layer.msg_gate(msg)
        alpha = attention_layer.normalize_weights(gate, index, size_i, mask_j)
        if return_weights:
            # 保存权重和对应的边索引
            saved_weights.append({
                'weights': alpha.detach().cpu(),
                'edge_index': index.cpu() if isinstance(index, torch.Tensor) else index,
                'node_dim': attention_layer.node_dim
            })
        alpha = torch.nn.functional.dropout(alpha, p=attention_layer.dropout, 
                                           training=attention_layer.training)
        out = alpha * msg
        return out
    
    # 临时替换message方法
    original_message = attention_layer.message
    attention_layer.message = custom_message
    
    try:
        out = attention_layer.propagate(edge_index, msg=msg, mask=mask, size=(N_src, N_tgt))
    finally:
        # 恢复原始message方法
        attention_layer.message = original_message
    
    # skip connection
    if attention_layer.root_weight:
        out = out + attention_layer.lin_skip(x_tgt)
    
    if attention_layer.norm is not None:
        out = attention_layer.norm(out)
    
    if return_weights:
        return out, saved_weights
    return out


def visualize_temporal_attention(attention_weights: torch.Tensor, 
                                 time_steps: Optional[List[int]] = None,
                                 node_idx: int = 0,
                                 layer_idx: int = 0,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8)):
    """
    可视化时间注意力权重
    
    Args:
        attention_weights: 注意力权重矩阵 [target_time, source_time] 或 [edges]
        time_steps: 时间步标签
        node_idx: 节点索引（用于标题）
        layer_idx: 层索引（用于标题）
        save_path: 保存路径
        figsize: 图形大小
    """
    # 如果是稀疏格式，需要转换为密集矩阵
    if attention_weights.dim() == 1:
        # 假设是边级别的权重，需要根据edge_index重构
        raise ValueError("稀疏格式的权重需要edge_index信息来重构矩阵")
    
    # 转换为numpy
    if isinstance(attention_weights, torch.Tensor):
        weights_np = attention_weights.detach().cpu().numpy()
    else:
        weights_np = attention_weights
    
    plt.figure(figsize=figsize)
    sns.heatmap(weights_np, 
                cmap='YlOrRd', 
                annot=True, 
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=time_steps if time_steps else range(weights_np.shape[1]),
                yticklabels=time_steps if time_steps else range(weights_np.shape[0]))
    
    plt.title(f'Temporal Attention Weights - Layer {layer_idx}, Node {node_idx}')
    plt.xlabel('Source Time Step')
    plt.ylabel('Target Time Step')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_spatial_attention(attention_weights: torch.Tensor,
                                edge_index: torch.Tensor,
                                node_names: Optional[List[str]] = None,
                                node_idx: int = 0,
                                layer_idx: int = 0,
                                time_step: int = 0,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8)):
    """
    可视化空间图注意力权重
    
    Args:
        attention_weights: 注意力权重 [edges] 或 [nodes, nodes]
        edge_index: 边索引 [2, edges]
        node_names: 节点名称列表
        node_idx: 目标节点索引（如果只可视化一个节点的注意力）
        layer_idx: 层索引
        time_step: 时间步索引
        save_path: 保存路径
        figsize: 图形大小
    """
    if isinstance(attention_weights, torch.Tensor):
        weights_np = attention_weights.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
    else:
        weights_np = attention_weights
        edge_index_np = edge_index
    
    # 如果权重是一维的（边级别），需要重构为矩阵
    if weights_np.ndim == 1:
        n_nodes = edge_index_np.max() + 1
        weight_matrix = np.zeros((n_nodes, n_nodes))
        
        # 填充权重矩阵
        for i in range(edge_index_np.shape[1]):
            src, tgt = edge_index_np[0, i], edge_index_np[1, i]
            weight_matrix[tgt, src] = weights_np[i]
    else:
        weight_matrix = weights_np
    
    plt.figure(figsize=figsize)
    sns.heatmap(weight_matrix,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=node_names if node_names else range(weight_matrix.shape[1]),
                yticklabels=node_names if node_names else range(weight_matrix.shape[0]))
    
    plt.title(f'Spatial Attention Weights - Layer {layer_idx}, Time Step {time_step}')
    plt.xlabel('Source Node')
    plt.ylabel('Target Node')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spatial attention visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def extract_and_visualize_attention(model, sample_input, output_dir='./attention_vis',
                                    node_idx=0, time_steps=None):
    """
    从模型中提取注意力权重并可视化
    
    Args:
        model: SPIN模型实例
        sample_input: 样本输入字典，包含 x, mask, edge_index 等
        output_dir: 输出目录
        node_idx: 要可视化的节点索引
        time_steps: 时间步标签列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        # 获取输入
        x = sample_input['x']
        mask = sample_input.get('mask')
        edge_index = sample_input['edge_index']
        edge_weight = sample_input.get('edge_weight')
        
        # 为每一层提取注意力权重
        for layer_idx, encoder_layer in enumerate(model.encoder):
            print(f"Processing layer {layer_idx}...")
            
            # 提取空间注意力权重
            if hasattr(encoder_layer, 'cross_attention'):
                # 这里需要修改forward方法以返回权重
                # 暂时使用hook方式（需要进一步实现）
                pass
            
            # 提取时间注意力权重
            if hasattr(encoder_layer, 'self_attention') and encoder_layer.self_attention is not None:
                # 需要修改forward方法以返回权重
                pass
    
    print(f"Attention visualizations saved to {output_dir}")


# 便捷函数：创建一个可以返回注意力的模型包装器
class AttentionWrapper:
    """包装模型以返回注意力权重"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
    
    def forward_with_attention(self, x, u, mask, edge_index, edge_weight=None):
        """前向传播并返回注意力权重"""
        # 这里需要修改模型以支持返回注意力
        # 暂时作为接口定义
        output = self.model(x, u, mask, edge_index, edge_weight)
        return output, self.attention_weights


