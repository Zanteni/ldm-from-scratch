import torch.nn as nn


class Sequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def append(self, layer):
        self.layers.append(layer)