"""SAC actor heads.

`DiagGaussianActor` is the non-recurrent actor. `RSACActor` and `AugRSACActor`
are recurrent variants whose only difference is which GRU implementation they
use internally — the Aug-GRU one is what survives ESP-DL quantization.

For ESP-DL deployment we only export the inference path: the SquashedNormal
distribution collapses to ``tanh(mu)`` because TanhTransform of the mean of a
Normal *is* the mean of the SquashedNormal.
"""

import math

import torch
import torch.nn as nn
from torch import distributions as pyd
import torch.nn.functional as F

from drl_quant.networks import utils


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        super().__init__(self.base_dist, [TanhTransform()])

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """Non-recurrent SAC actor."""

    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.action_dim = action_dim

        self.sac_diag_gaussian_actor = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size2']),
            nn.ReLU(),
            nn.Linear(params['hidden_size2'], action_dim * 2),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        # Inference path only: take the mu half and tanh it.
        mu = self.sac_diag_gaussian_actor(obs.float())[0][0:self.action_dim]
        return mu.tanh()


class RSACActor(nn.Module):
    """Recurrent SAC actor using the standard PyTorch `nn.GRU`."""

    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.action_dim = action_dim

        self.sac_diag_gaussian_actor = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size2']),
            nn.ReLU(),
            nn.Linear(params['hidden_size2'], action_dim * 2),
        )

        self.rl_gru = nn.GRU(
            hidden_size=params['rnn_hidden_size'],
            input_size=state_dim,
            num_layers=params['rnn_layers'],
            batch_first=True,
            bias=False,
        )

        self.apply(utils.weight_init)

    def forward(self, state, h_t):
        rnn_input = state.unsqueeze(dim=0)
        out, h_t = self.rl_gru(rnn_input, h_t)
        full_input = torch.cat(
            (rnn_input.squeeze(dim=0).squeeze(dim=0), out.squeeze(0).squeeze(0))
        )
        mu = self.sac_diag_gaussian_actor(full_input)[0:self.action_dim]
        return mu.tanh(), h_t


class AugRSACActor(nn.Module):
    """Recurrent SAC actor using `GRUAug` — the only variant that survives
    ESP-DL quantization. Has the same `state_dict` shape as `RSACActor` so
    weights from a trained `RSACActor` can be transplanted (see
    `drl_quant.onnx_export.export_aug_actor`)."""

    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.action_dim = action_dim

        self.sac_diag_gaussian_actor = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size2']),
            nn.ReLU(),
            nn.Linear(params['hidden_size2'], action_dim * 2),
        )

        self.rl_gru = nn.GRU(
            hidden_size=params['rnn_hidden_size'],
            input_size=state_dim,
            num_layers=params['rnn_layers'],
            batch_first=True,
            bias=False,
        )

        self.apply(utils.weight_init)

    def forward(self, state, h_t):
        rnn_input = state
        out, h_t = self.rl_gru(rnn_input, h_t)
        full_input = torch.cat(
            (rnn_input.squeeze(dim=0).squeeze(dim=0), out.squeeze(0).squeeze(0))
        )
        mu = self.sac_diag_gaussian_actor(full_input)[0:self.action_dim]
        return mu.tanh(), h_t
