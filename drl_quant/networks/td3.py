"""TD3 actor heads.

The same three-way split as SAC: non-recurrent (`TD3Actor`), recurrent with
native `nn.GRU` (`RTD3Actor`), and recurrent with the export-friendly
`GRUAug` (`AugRTD3Actor`).

The non-recurrent variant carries a `QuantStub` because we historically also
ran torch.quantization on it for benchmarking; ESP-DL quantization is a
separate path and does not depend on the stub.
"""

import torch
from torch import nn

from drl_quant.networks import utils


class TD3Actor(nn.Module):
    """Non-recurrent TD3 actor."""

    def __init__(self, state_dim, action_dim, params):
        super().__init__()

        self.td3_actor_net = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size2']),
            nn.ReLU(),
            nn.Linear(params['hidden_size2'], action_dim),
            nn.Tanh(),
        )
        self.quant = torch.quantization.QuantStub()

        if params.get('orthogonal_init', False):
            self.apply(utils.weight_init)

    def forward(self, state):
        # QuantStub is a no-op outside torch.quantization tracing; left in
        # so the same module works for both ESP-DL and torch quant flows.
        state = self.quant(state)
        return self.td3_actor_net(state)


class RTD3Actor(nn.Module):
    """Recurrent TD3 actor using the standard PyTorch `nn.GRU`."""

    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.params = params

        self.td3_actor_net = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size2']),
            nn.ReLU(),
            nn.Linear(params['hidden_size2'], action_dim),
            nn.Tanh(),
        )
        self.quant = torch.quantization.QuantStub()

        # h_t is passed in/out instead of being a module attribute because
        # ONNX cannot capture instance state across calls.
        self.rl_gru = nn.GRU(
            hidden_size=params['rnn_hidden_size'],
            input_size=state_dim,
            num_layers=params['rnn_layers'],
            batch_first=True,
            bias=False,
        )

        if params.get('orthogonal_init', False):
            self.apply(utils.weight_init)

    def forward(self, state, h_t):
        if h_t is None:
            h_t = torch.zeros(self.params['rnn_layers'], 1, self.params['rnn_hidden_size'])
        rnn_input = state.unsqueeze(dim=0)
        out, h_t = self.rl_gru(rnn_input, h_t)
        full_input = torch.cat(
            (rnn_input.squeeze(dim=0).squeeze(dim=0), out.squeeze(0).squeeze(0))
        )
        return self.td3_actor_net(full_input), h_t


class AugRTD3Actor(nn.Module):
    """Recurrent TD3 actor using `GRUAug` — ESP-DL-friendly variant.

    Has the same `state_dict` shape as `RTD3Actor` so weights from a trained
    `RTD3Actor` can be transplanted directly.
    """

    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.params = params

        self.td3_actor_net = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size2']),
            nn.ReLU(),
            nn.Linear(params['hidden_size2'], action_dim),
            nn.Tanh(),
        )

        self.rl_gru = nn.GRU(
            hidden_size=params['rnn_hidden_size'],
            input_size=state_dim,
            num_layers=params['rnn_layers'],
            batch_first=True,
            bias=False,
        )

        if params.get('orthogonal_init', False):
            self.apply(utils.weight_init)

    def forward(self, state, h_t):
        rnn_input = state
        out, h_t = self.rl_gru(rnn_input, h_t)
        full_input = torch.cat(
            (rnn_input.squeeze(dim=0).squeeze(dim=0), out.squeeze(0).squeeze(0))
        )
        return self.td3_actor_net(full_input), h_t
