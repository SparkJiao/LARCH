import torch
from torch import nn, Tensor
from torch.nn import Linear

ACT_FN = {
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh
}


class FuseLayer(nn.Module):
    def __init__(self, input_size, gate_type: int = 0, act_fn: str = "gelu", dropout=0.1):
        super().__init__()

        self.f = nn.Linear(input_size * 4, input_size)
        if gate_type == 0:
            self.g = nn.Linear(input_size * 4, input_size)
        elif gate_type == 1:
            self.g = nn.Linear(input_size * 4, 1)
        else:
            raise ValueError()
        self.f_act_fn = ACT_FN[act_fn]()
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

    def forward(self, x, y):
        z = self.dropout(torch.cat([x, y, x - y, x * y], dim=-1))
        f = self.f_act_fn(self.f(z))
        g = torch.sigmoid(self.g(z))
        res = g * f + (1 - g) * x
        return res


def weighted_average(w: Linear, x: Tensor, mask: Tensor):
    scores = w(x).squeeze(-1) + (1 - mask) * -10000.0
    alpha = torch.softmax(scores, dim=-1)
    ...
