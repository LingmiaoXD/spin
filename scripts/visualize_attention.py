"""
SPIN模型注意力权重可视化脚本

使用方法:
    python scripts/visualize_attention.py --checkpoint <模型checkpoint路径> --data_dir <数据目录>
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from spin.layers.additive_attention import AdditiveAttention
from spin.models.spin import SPINModel


class AttentionWeightExtractor:
    """提取注意力权重的辅助类"""
    
    def __init__(self):
        self.attention_weights = {}
        self.hooks = []
    
    def hook_message_function(self, layer_name, layer_idx, attn_type='spatial'):
        """为message函数创建hook以捕获注意力权重"""
        def hook_fn(module, input, output):
            # 获取message函数中的权重
            # 注意：这需要在message函数内部添加hook
            key = f"{layer_name}_layer{layer_idx}_{attn_type}"
            # 权重会在message函数中计算，我们需要在那里捕获
            pass
        return hook_fn
    
    def extract_attention_from_forward(self, module, input, output, layer_name):
        """从前向传播中提取注意力权重"""
        # 这个方法需要在message函数中手动保存权重
        pass


def modify_additive_attention_for_extraction():
    """
    临时修改AdditiveAttention类以支持权重提取
    这个函数会创建一个临时的修改版本
    """
    original_message = AdditiveAttention.message
    
    def message_with_extraction(self, msg_j, msg_i, index, size_i, mask_j=None):
        """带权重提取的message函数"""
        msg = self.msg_nn(msg_j + msg_i)
        gate = self.msg_gate(msg)
        alpha = self.normalize_weights(gate, index, size_i, mask_j)
        
        # 保存权重（如果设置了提取标志）
        if hasattr(self, '_extract_weights') and self._extract_weights:
            if not hasattr(self, '_saved_weights'):
                self._saved_weights = []
            self._saved_weights.append({
                'weights': alpha.detach().cpu().clone(),
                'index': index.detach().cpu().clone() if isinstance(index, torch.Tensor) else index,
                'size_i': size_i
            })
        
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        out = alpha * msg
        return out
    
    return message_with_extraction


def extract_attention_weights_simple(model, x, mask, edge_index, 
                                     batch_idx=0, node_idx=0, 
                                     n_time_steps=None):
    """
    简单方法：通过临时修改模型的message方法来提取注意力权重
    
    Args:
        model: SPIN模型
        x: 输入数据 [B, T, N, C]
        mask: mask [B, T, N, C]
        edge_index: 边索引
        batch_idx: 批次索引
        node_idx: 节点索引
        n_time_steps: 时间步数
    
    Returns:
        spatial_weights: 空间注意力权重字典
        temporal_weights: 时间注意力权重字典
    """
    model.eval()
    
    # 保存原始的message方法
    saved_methods = {}
    
    # 为所有注意力层启用权重提取
    def enable_weight_extraction(module):
        if isinstance(module, AdditiveAttention):
            module._extract_weights = True
            module._saved_weights = []
            # 临时替换message方法
            original_msg = module.message
            saved_methods[id(module)] = original_msg
            
            def message_with_extraction(self, msg_j, msg_i, index, size_i, mask_j=None):
                msg = self.msg_nn(msg_j + msg_i)
                gate = self.msg_gate(msg)
                alpha = self.normalize_weights(gate, index, size_i, mask_j)
                
                # 保存权重
                if not hasattr(self, '_saved_weights'):
                    self._saved_weights = []
                self._saved_weights.append({
                    'weights': alpha.detach().cpu().clone(),
                    'index': index.detach().cpu().clone() if isinstance(index, torch.Tensor) else index,
                    'size_i': size_i
                })
                
                alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)
                out = alpha * msg
                return out
            
            # 绑定self到新方法
            import types
            module.message = types.MethodType(message_with_extraction, module)
    
    # 遍历所有层并启用权重提取
    for layer in model.encoder:
        if hasattr(layer, 'cross_attention'):
            enable_weight_extraction(layer.cross_attention)
        if hasattr(layer, 'self_attention') and layer.self_attention is not None:
            enable_weight_extraction(layer.self_attention)
    
    # 执行前向传播
    with torch.no_grad():
        output = model(x, u=None, mask=mask, edge_index=edge_index)
    
    # 收集权重
    spatial_weights = {}
    temporal_weights = {}
    
    for layer_idx, layer in enumerate(model.encoder):
        if hasattr(layer, 'cross_attention') and hasattr(layer.cross_attention, '_saved_weights'):
            spatial_weights[f'layer_{layer_idx}'] = layer.cross_attention._saved_weights
        if hasattr(layer, 'self_attention') and layer.self_attention is not None:
            if hasattr(layer.self_attention, '_saved_weights'):
                temporal_weights[f'layer_{layer_idx}'] = layer.self_attention._saved_weights
    
    # 恢复原始方法
    for layer in model.encoder:
        if hasattr(layer, 'cross_attention'):
            module_id = id(layer.cross_attention)
            if module_id in saved_methods:
                layer.cross_attention.message = saved_methods[module_id]
                layer.cross_attention._extract_weights = False
        if hasattr(layer, 'self_attention') and layer.self_attention is not None:
            module_id = id(layer.self_attention)
            if module_id in saved_methods:
                layer.self_attention.message = saved_methods[module_id]
                layer.self_attention._extract_weights = False
    
    return spatial_weights, temporal_weights


def reconstruct_temporal_attention_matrix(saved_weights, n_time_steps):
    """
    从保存的权重重构时间注意力矩阵
    
    Args:
        saved_weights: 保存的权重列表（来自message函数）
        n_time_steps: 时间步数
    
    Returns:
        attention_matrix: [target_time, source_time] 的注意力矩阵
    """
    if not saved_weights:
        return None
    
    # 假设所有时间步都连接，使用最后一个权重项
    # 注意：这里需要根据实际的edge_index来重构
    attention_matrix = np.zeros((n_time_steps, n_time_steps))
    
    # 如果saved_weights中有多个项，通常是批处理的结果
    # 我们需要找到对应的时间步连接
    weights_data = saved_weights[-1]  # 使用最后一个（通常是完整的前向传播）
    weights = weights_data['weights'].numpy()
    indices = weights_data['index'].numpy() if isinstance(weights_data['index'], torch.Tensor) else weights_data['index']
    
    # 这里需要知道edge_index的结构来正确映射
    # 暂时假设weights和indices对应边连接
    # 实际使用时需要传入edge_index信息
    
    return attention_matrix


def visualize_temporal_attention_matrix(attention_matrix, layer_idx, 
                                        save_path=None, figsize=(10, 8)):
    """可视化时间注意力矩阵"""
    plt.figure(figsize=figsize)
    sns.heatmap(attention_matrix,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'},
                square=True)
    plt.title(f'Temporal Attention Weights - Layer {layer_idx}')
    plt.xlabel('Source Time Step')
    plt.ylabel('Target Time Step')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_spatial_attention_edges(weights_list, edge_index, layer_idx,
                                      node_idx=0, save_path=None, figsize=(12, 8)):
    """可视化空间注意力的边权重"""
    if not weights_list:
        print(f"No weights found for layer {layer_idx}")
        return
    
    # 使用最后一个权重项（完整前向传播的结果）
    weights_data = weights_list[-1]
    weights = weights_data['weights'].numpy().flatten()
    
    if isinstance(edge_index, torch.Tensor):
        edge_index_np = edge_index.detach().cpu().numpy()
    else:
        edge_index_np = edge_index
    
    # 创建权重字典
    edge_weights_dict = {}
    n_edges = min(len(weights), edge_index_np.shape[1])
    
    for i in range(n_edges):
        src, tgt = edge_index_np[0, i], edge_index_np[1, i]
        edge_weights_dict[(src, tgt)] = weights[i]
    
    # 可视化（这里可以用图可视化库如networkx）
    print(f"Layer {layer_idx} - Edge attention weights (first 10):")
    for i, ((src, tgt), w) in enumerate(list(edge_weights_dict.items())[:10]):
        print(f"  Edge ({src} -> {tgt}): {w:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize SPIN attention weights')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to sample data file')
    parser.add_argument('--output_dir', type=str, default='./attention_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--node_idx', type=int, default=0,
                       help='Node index to visualize')
    parser.add_argument('--batch_idx', type=int, default=0,
                       help='Batch index to use')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 这里需要根据实际的checkpoint结构来加载模型
    # model = SPINModel(...)
    # model.load_state_dict(checkpoint['state_dict'])
    
    print("Model loaded. Note: This script requires model instantiation code.")
    print("Please adapt the model loading part according to your checkpoint format.")
    
    # 示例使用：
    # spatial_weights, temporal_weights = extract_attention_weights_simple(
    #     model, x, mask, edge_index, 
    #     batch_idx=args.batch_idx, 
    #     node_idx=args.node_idx
    # )
    
    print(f"Visualizations will be saved to {output_dir}")


if __name__ == '__main__':
    main()


