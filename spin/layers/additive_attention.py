from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import LayerNorm, functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter
from torch_scatter.utils import broadcast
from tsl.nn.blocks.encoders import MLP
from tsl.nn.functional import sparse_softmax


class AdditiveAttention(MessagePassing):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 dim: int = -2,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=dim, **kwargs)

        self.output_size = output_size
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size

        self.msg_size = msg_size or self.output_size
        self.msg_layers = msg_layers

        assert reweight in ['softmax', 'l1', None]
        self.reweight = reweight

        self.root_weight = root_weight
        self.dropout = dropout

        # key bias is discarded in softmax
        self.lin_src = Linear(self.src_size, self.output_size,
                              weight_initializer='glorot',
                              bias_initializer='zeros')
        self.lin_tgt = Linear(self.tgt_size, self.output_size,
                              weight_initializer='glorot', bias=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size,
                                   bias=False)
        else:
            self.register_parameter('lin_skip', None)

        self.msg_nn = nn.Sequential(
            nn.PReLU(init=0.2),
            MLP(self.output_size, self.msg_size, self.output_size,
                n_layers=self.msg_layers, dropout=self.dropout,
                activation='prelu')
        )

        if self.reweight == 'softmax':
            self.msg_gate = nn.Linear(self.output_size, 1, bias=False)
        else:
            self.msg_gate = nn.Sequential(nn.Linear(self.output_size, 1),
                                          nn.Sigmoid())

        if norm:
            self.norm = LayerNorm(self.output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_tgt.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()

    def forward(self, x: PairTensor, edge_index: Adj, mask: OptTensor = None):
        # if query/key not provided, defaults to x (e.g., for self-attention)
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        N_src, N_tgt = x_src.size(self.node_dim), x_tgt.size(self.node_dim)

        msg_src = self.lin_src(x_src)
        msg_tgt = self.lin_tgt(x_tgt)

        msg = (msg_src, msg_tgt)

        # propagate_type: (msg: PairTensor, mask: OptTensor)
        out = self.propagate(edge_index, msg=msg, mask=mask,
                             size=(N_src, N_tgt))

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def normalize_weights(self, weights, index, num_nodes, mask=None):
        # mask weights - 避免就地修改
        if mask is not None:
            fill_value = float("-inf") if self.reweight == 'softmax' else 0.
            weights = weights.clone().masked_fill(torch.logical_not(mask), fill_value)
        # eventually reweight
        if self.reweight == 'l1':
            expanded_index = broadcast(index, weights, self.node_dim)
            weights_sum = scatter(weights, expanded_index, self.node_dim,
                                  dim_size=num_nodes, reduce='sum')
            weights_sum = weights_sum.index_select(self.node_dim, index)
            weights = weights / (weights_sum + 1e-5)
        elif self.reweight == 'softmax':
            weights = sparse_softmax(weights, index, num_nodes=num_nodes,
                                     dim=self.node_dim)
        return weights

    def message(self, msg_j: Tensor, msg_i: Tensor, index, size_i,
                mask_j: OptTensor = None) -> Tensor:
        msg = self.msg_nn(msg_j + msg_i)
        gate = self.msg_gate(msg)
        alpha = self.normalize_weights(gate, index, size_i, mask_j)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = alpha * msg
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.output_size}, '
                f'dim={self.node_dim}, '
                f'root_weight={self.root_weight})')


class TemporalAdditiveAttention(AdditiveAttention):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 norm: bool = True,
                 dropout: float = 0.0,
                 temporal_distance_bias: bool = True,
                 temporal_bias_scale: float = 1.0,
                 max_temporal_distance: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('dim', 1)
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         msg_size=msg_size,
                         msg_layers=msg_layers,
                         root_weight=root_weight,
                         reweight=reweight,
                         dropout=dropout,
                         norm=norm,
                         **kwargs)
        self.temporal_distance_bias = temporal_distance_bias
        self.temporal_bias_scale = temporal_bias_scale
        self.max_temporal_distance = max_temporal_distance

    def forward(self, x: PairTensor, mask: OptTensor = None,
                temporal_mask: OptTensor = None,
                causal_lag: Optional[int] = None):
        # x: [b s * c]    query: [b l * c]    key: [b s * c]
        # mask: [b s * c]    temporal_mask: [l s]
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        l, s = x_tgt.size(self.node_dim), x_src.size(self.node_dim)
        i = torch.arange(l, dtype=torch.long, device=x_src.device)
        j = torch.arange(s, dtype=torch.long, device=x_src.device)

        # compute temporal index, from j to i
        if temporal_mask is None and isinstance(causal_lag, int):
            temporal_mask = tuple(torch.tril_indices(l, l, offset=-causal_lag,
                                                     device=x_src.device))
        if temporal_mask is not None:
            assert temporal_mask.size() == (l, s)
            # 使用兼容的方式创建网格
            try:
                # PyTorch >= 1.10.0 支持 indexing 参数
                i_grid, j_grid = torch.meshgrid([i, j], indexing='ij')
            except TypeError:
                # 旧版本 PyTorch，需要转置
                j_grid, i_grid = torch.meshgrid([j, i])
            edge_index = torch.stack((j_grid[temporal_mask], i_grid[temporal_mask]))
        else:
            # 如果没有提供temporal_mask，根据max_temporal_distance限制连接范围
            if self.max_temporal_distance is not None:
                # 只连接距离在max_temporal_distance内的时间步
                try:
                    # PyTorch >= 1.10.0 支持 indexing 参数
                    i_grid, j_grid = torch.meshgrid([i, j], indexing='ij')
                except TypeError:
                    # 旧版本 PyTorch，需要转置
                    j_grid, i_grid = torch.meshgrid([j, i])
                distances = torch.abs(i_grid - j_grid)
                mask = distances <= self.max_temporal_distance
                edge_index = torch.stack((j_grid[mask], i_grid[mask]))
            else:
                # 如果没有限制，使用全连接（保持向后兼容）
                edge_index = torch.cartesian_prod(j, i).T

        # 计算时间距离偏置：距离越短，偏置越大
        if self.temporal_distance_bias:
            # edge_index: [2, num_edges], 第一行是源时间步j，第二行是目标时间步i
            temporal_distances = torch.abs(edge_index[0] - edge_index[1]).float()
            # 将距离转换为偏置：距离越短，偏置越大（使用负指数或倒数）
            # 使用 exp(-scale * distance) 使得距离越短，偏置越大
            temporal_bias = torch.exp(-self.temporal_bias_scale * temporal_distances)
            # 存储时间偏置以供message方法使用
            self._temporal_bias = temporal_bias
        else:
            self._temporal_bias = None

        return super(TemporalAdditiveAttention, self).forward(x, edge_index,
                                                              mask=mask)

    def message(self, msg_j: Tensor, msg_i: Tensor, index, size_i,
                mask_j: OptTensor = None) -> Tensor:
        msg = self.msg_nn(msg_j + msg_i)
        gate = self.msg_gate(msg)
        
        # 如果有时间距离偏置，将其应用到注意力权重上
        if self.temporal_distance_bias and hasattr(self, '_temporal_bias'):
            # temporal_bias: [num_edges], gate: [num_edges, 1] 或 [num_edges]
            # 确保维度匹配
            temporal_bias = self._temporal_bias
            if gate.dim() > 1:
                # gate: [num_edges, 1]
                temporal_bias = temporal_bias.unsqueeze(-1)
            
            # 将时间偏置加到gate上
            # 对于softmax归一化，加性偏置会在归一化时起作用
            # 对于其他归一化方式，也使用加性偏置以保持一致性
            gate = gate + temporal_bias
        
        alpha = self.normalize_weights(gate, index, size_i, mask_j)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = alpha * msg
        return out
