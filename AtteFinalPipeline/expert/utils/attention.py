import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SelfAttention", "TemporalAttention"]


def conv1d(
    ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False
):
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return conv


class SelfAttention(nn.Module):
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()
        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels // div)
            self.key = conv1d(n_channels, n_channels // div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, return_attention=False):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = (self.query(x), self.key(x), self.value(x))
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        if return_attention:
            return (o.view(*size).contiguous(), beta)
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return (context, weights_att.permute(1, 0, 2))
