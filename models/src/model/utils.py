from torch import nn
import torch


class FM(nn.Module):
    def __init__(self, n=None, k=None):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)
        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out
