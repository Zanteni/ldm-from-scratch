import torch
import torch.nn as nn


class Dropout(nn.Module):
    """
    Inverted dropout: randomly zeroes elements with probability p during
    training, and scales survivors by 1/(1-p) to keep the expected value
    of the output equal to the input. This scaling means inference needs
    no special handling — eval mode just passes input through unchanged.

    Derivation: for X ~ Uniform[0, 1), P(X > p) = 1 - p, so comparing a
    random uniform tensor against p keeps each element with probability
    (1-p) and drops it with probability p, matching the target drop rate.
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p <= 1, f"p must be a probability between 0 and 1, got {p}."
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        mask = (torch.rand_like(x) > self.p).float()
        return mask * x / (1 - self.p)