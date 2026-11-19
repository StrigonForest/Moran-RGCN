from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_geometric
from torch import Tensor
from torch.nn import Parameter
from torch_geometric import is_compiling
from torch_geometric.index import index2ptr
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    pyg_lib,
)
from torch_geometric.utils import index_sort


class MoranRGCNConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int,
            aggr: str = 'mean',
            root_weight: bool = True,
            is_sorted: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self.weight = Parameter(
            torch.empty(num_relations, in_channels[0], out_channels))
        self.register_parameter('comp', None)

        if root_weight:
            self.root = Parameter(torch.empty(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_type: OptTensor = None,
                edge_weight: Tensor | None = None
    ):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = x
        x_r: Tensor = x_l

        size = (x_l.size(0), x_r.size(0))
        assert edge_type is not None

        if not self.is_sorted:
            if (edge_type[1:] < edge_type[:-1]).any():
                edge_type, perm = index_sort(
                    edge_type, max_value=self.num_relations)
                edge_index = edge_index[:, perm]
                if edge_weight is not None:
                    edge_weight = edge_weight[perm]

        edge_type_ptr = index2ptr(edge_type, self.num_relations)

        out = self.propagate(
            edge_index,
            x=x_l,
            edge_type=edge_type,
            edge_type_ptr=edge_type_ptr,
            edge_weight=edge_weight,
            size=size
        )

        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
            self,
            x_j: Tensor,
            edge_type_ptr: OptTensor,
            edge_type: Tensor,
            edge_weight: Optional[Tensor],
    ) -> Tensor:
        if (
                torch_geometric.typing.WITH_SEGMM
                and not is_compiling()
                and edge_type_ptr is not None
        ):
            if edge_weight is not None:
                x_j = x_j * edge_weight.unsqueeze(-1)
            # TODO Re-weight according to edge type degree for `aggr=mean`.
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)

        # Generic path: per-edge W_r
        W = self.weight.index_select(0, edge_type)  # [E, F_in, F_out]
        msg = torch.bmm(x_j.unsqueeze(1), W).squeeze(1)  # [E, F_out]
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        return msg


class MoranRGCN(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_relations: int,
            in_feats: int,
            hidden_feats: int,
            out_feats: int,
            dropout: float,
    ):
        super(MoranRGCN, self).__init__()
        self.num_layers = num_layers
        self.neigh1_layers = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.neigh1_layers.append(
            MoranRGCNConv(in_feats, hidden_feats, num_relations=num_relations, root_weight=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.neigh1_layers.append(
                MoranRGCNConv(hidden_feats, hidden_feats, num_relations=num_relations, root_weight=True))

        # Output layer
        self.neigh1_layers.append(
            MoranRGCNConv(hidden_feats, out_feats, num_relations=num_relations, root_weight=True))

        self.reset_parameters()

    def reset_parameters(self):
        for _, neigh1_layer in enumerate(self.neigh1_layers):
            neigh1_layer.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_weight):
        for i in range(self.num_layers - 1):
            x = self.neigh1_layers[i](x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer
        x = self.neigh1_layers[-1](x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight)
        return x