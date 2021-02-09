import torch
from torch import nn
from torch.nn import functional

# from .gat_layer import GraphAttentionLayer
from .gat_layer import GATLayer
from constants import GAT_LAYER

# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nfeat_out, dropout, alpha, nheads):
#         super(GAT, self).__init__()
#         self.dropout = dropout
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
#                            range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         self.out_att = GraphAttentionLayer(nhid * nheads, nfeat_out, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         result = functional.elu(self.out_att(x, adj))
#         return result


class GAT(nn.Module):
    def __init__(self, input_dim, num_attn_heads, dropout, alpha):
        super(GAT, self).__init__()

        self.attentions = nn.ModuleList()
        for i in range(GAT_LAYER - 1):
            self.attentions.append(GATLayer(input_dim, num_attn_heads, dropout, alpha, concat=True))
        self.attentions.append(GATLayer(input_dim, num_attn_heads, dropout, alpha, concat=False))

    def forward(self, x, adj):
        for i in range(GAT_LAYER):
            x = self.attentions[i](x, adj)
        return x
