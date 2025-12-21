from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.typing import OptTensor


class LSTMModel(nn.Module):
    """
    LSTM模型用于时间序列插补
    
    Args:
        input_size: 输入特征维度
        hidden_size: LSTM隐藏层维度
        n_nodes: 节点数量
        n_layers: LSTM层数
        dropout: Dropout比率
        bidirectional: 是否使用双向LSTM
        output_size: 输出特征维度（默认与input_size相同）
    """

    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 output_size: Optional[int] = None):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        output_size = output_size or input_size
        self.output_size = output_size
        
        # 对于每个节点，使用独立的LSTM处理时间序列
        # 输入形状: [batch, seq_len, n_nodes, input_size]
        # 我们需要将n_nodes和input_size合并，然后对每个节点分别处理
        
        # 使用LSTM处理时间序列
        # 输入: [batch * n_nodes, seq_len, input_size]
        # 输出: [batch * n_nodes, seq_len, hidden_size * (2 if bidirectional else 1)]
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(lstm_output_size, output_size)
        
        # 可选的层归一化
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, 
                x: Tensor, 
                u: OptTensor = None,
                mask: Tensor = None,
                edge_index: OptTensor = None,
                edge_weight: OptTensor = None,
                node_index: OptTensor = None,
                target_nodes: OptTensor = None):
        """
        前向传播
        
        Args:
            x: 输入数据 [batch, seq_len, n_nodes, input_size]
            u: 时间编码（可选）[batch, seq_len, u_size]
            mask: 掩码 [batch, seq_len, n_nodes, input_size] 或 [batch, seq_len, n_nodes, 1]
            edge_index: 边索引（LSTM不使用，但保持接口一致）
            edge_weight: 边权重（LSTM不使用，但保持接口一致）
            node_index: 节点索引（可选）
            target_nodes: 目标节点索引（可选）
        
        Returns:
            x_hat: 插补结果 [batch, seq_len, n_nodes, output_size]
            predictions: 中间预测列表（用于多任务学习，LSTM返回空列表）
        """
        batch_size, seq_len, n_nodes, input_size = x.shape
        
        # 处理mask
        if mask is not None:
            # 如果mask的最后一维是1，扩展到input_size
            if mask.shape[-1] == 1:
                mask = mask.expand(-1, -1, -1, input_size)
            # 将缺失值置为0
            x_masked = x * mask
        else:
            x_masked = x
        
        # 如果指定了target_nodes，只处理目标节点
        need_padding = False
        if target_nodes is not None:
            if isinstance(target_nodes, slice):
                # slice(None)表示所有节点，不需要特殊处理
                if target_nodes != slice(None):
                    x_masked = x_masked[..., target_nodes, :]
                    n_nodes = x_masked.shape[2]
            else:
                # 具体的节点索引列表
                x_masked = x_masked[..., target_nodes, :]
                n_nodes = x_masked.shape[2]
                need_padding = True
                original_target_nodes = target_nodes
        
        # 重新排列维度: [batch, seq_len, n_nodes, input_size] -> [batch * n_nodes, seq_len, input_size]
        x_reshaped = x_masked.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_reshaped.view(batch_size * n_nodes, seq_len, input_size)
        
        # LSTM前向传播
        # output: [batch * n_nodes, seq_len, hidden_size * (2 if bidirectional else 1)]
        lstm_out, _ = self.lstm(x_reshaped)
        
        # 投影到输出维度
        # output: [batch * n_nodes, seq_len, output_size]
        output = self.output_proj(lstm_out)
        output = self.layer_norm(output)
        
        # 重新排列维度: [batch * n_nodes, seq_len, output_size] -> [batch, seq_len, n_nodes, output_size]
        output = output.view(batch_size, n_nodes, seq_len, self.output_size)
        output = output.permute(0, 2, 1, 3).contiguous()
        
        # 如果指定了具体的target_nodes索引，需要将输出填充回原始形状
        if need_padding:
            full_output = torch.zeros(
                batch_size, seq_len, self.n_nodes, self.output_size,
                device=output.device, dtype=output.dtype
            )
            full_output[..., original_target_nodes, :] = output
            output = full_output
        
        # LSTM模型不返回中间预测，只返回最终插补结果
        return output, []

    @staticmethod
    def add_model_specific_args(parser):
        """添加模型特定的命令行参数"""
        parser.add_argument('--hidden-size', type=int, default=64,
                           help='LSTM隐藏层维度')
        parser.add_argument('--n-layers', type=int, default=2,
                           help='LSTM层数')
        parser.add_argument('--dropout', type=float, default=0.1,
                           help='Dropout比率')
        parser.add_argument('--bidirectional', type=bool, default=False,
                           help='是否使用双向LSTM')
        parser.add_argument('--output-size', type=int, default=None,
                           help='输出特征维度（默认与input_size相同）')
        return parser

