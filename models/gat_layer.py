import torch
import torch.nn.functional as F
from torch import nn


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, residual=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input: torch.Tensor, adj):
        # Too harsh to use the same dropout. TODO add another dropout
        # input = F.dropout(input, self.dropout, training=self.training)

        seq = input.transpose(0, 1).unsqueeze(0)
        seq_fts = self.seq_transformation(seq)

        f_1 = self.f_1(seq_fts)
        f_2 = self.f_2(seq_fts)
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)

        seq_fts = F.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1), self.dropout, training=self.training)
        coefs = F.dropout(coefs, self.dropout, training=self.training)

        ret = torch.mm(coefs, seq_fts) + self.bias

        if self.residual:
            if seq.size()[-1] != ret.size()[-1]:
                ret += torch.transpose(self.proj_residual(seq).squeeze(0), 0, 1)
            else:
                ret += input

        if self.concat:
            return F.elu(ret)
        else:
            return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, head_num, dropout, alpha, concat=True, residual=False):
        super().__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        self.n_head = head_num
        self.d_head = in_features // self.n_head
        self.d_model = self.n_head * self.d_head
        self.seq_transform = nn.Linear(in_features, self.d_model, bias=False)

        self.attn_transform1 = nn.Linear(self.d_head, 1, bias=False)
        self.attn_transform2 = nn.Linear(self.d_head, 1, bias=False)

        self.attn_output = nn.Linear(self.d_model, self.in_features)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input: torch.Tensor, adj):
        # Too harsh to use the same dropout. TODO add another dropout
        # input = F.dropout(input, self.dropout, training=self.training)

        seq_len = input.size(0)

        seq = self.seq_transform(input).view(seq_len, self.n_head, self.d_head)

        # (n_head, seq_len)
        score1 = self.attn_transform1(seq).squeeze(-1).transpose(0, 1).contiguous()
        score2 = self.attn_transform2(seq).squeeze(-1).transpose(0, 1).contiguous()

        scores = score1[:, :, None] + score2[:, None, :]

        # seq1 = seq.unsqueeze(1).expand(-1, seq_len, -1).reshape(seq_len, seq_len, self.n_head, self.d_head)
        # seq2 = seq.unsqueeze(0).expand(seq_len, -1, -1).reshape(seq_len, seq_len, self.n_head, self.d_head)
        #
        # seq_mat = torch.cat([seq1, seq2], dim=-1).reshape(seq_len * seq_len, self.n_head, self.d_head * 2)
        # scores = self.attn_transform(seq_mat).view(seq_len, seq_len, self.n_head).permute(2, 0, 1)
        scores = self.leakyrelu(scores)
        alpha = torch.softmax(scores + (1 - adj).unsqueeze(0) * -65500.0, dim=-1)
        alpha = self.dropout(alpha)

        # seq = seq.view(seq_len, self.n_head, self.d_head)
        hidden = torch.einsum("hij,jhd->ihd", alpha, seq).reshape(seq_len, self.d_model)
        hidden = self.attn_output(hidden)

        if self.residual:
            hidden = hidden + input

        if self.concat:
            return F.elu(hidden)
        else:
            return hidden

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

