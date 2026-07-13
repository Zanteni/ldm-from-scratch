import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(in_features, out_features)
            / math.sqrt(in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = x @ self.weight

        if self.bias is not None:
            y = y + self.bias

        return y