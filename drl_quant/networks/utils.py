from torch import nn


def weight_init(m):
    """Orthogonal init for Linear layers, zero bias."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
