"""Thin wrappers around the standard PyTorch RNN ops.

`GruNet` is used to load weights from the upstream training repo's TorchScript
GRU into a plain `nn.GRU`, so they can be transplanted into either the native-
GRU exporter or the Aug-GRU exporter.
"""

import torch
import torch.nn as nn


class GruNet(nn.Module):
    """An ``nn.GRU`` wrapper that takes ``(x, h_t)`` and returns ``(out, h_t)``.

    ``out`` is squeezed to a 1-D feature vector for direct concat with the
    observation by the actor head. Defaults match the C++ training repo
    (3 layers, ``bias=False``); pass ``params={'bias': True}`` to support
    Python-trained GRUs that use the ``nn.GRU`` default of ``bias=True``.
    """

    def __init__(self, obs_size, params):
        super().__init__()
        self.rnn_hidden_size = params['rnn_hidden_size']
        self.rnn_layers = params['rnn_layers']

        self.rl_gru = nn.GRU(
            hidden_size=params['rnn_hidden_size'],
            input_size=obs_size,
            num_layers=params['rnn_layers'],
            batch_first=True,
            bias=params.get('bias', False),
        )

    def forward(self, x, h_t):
        out, h_t = self.rl_gru(x, h_t)
        return out.squeeze(0).squeeze(0), h_t


class GruNetWHT(nn.Module):
    """Same as `GruNet` but does not pass `h_t` through the forward signature.

    Kept around to demonstrate that hidden-state-less GRU export does NOT work
    on ESP-DL. Don't use this for anything you actually want to deploy.
    """

    def __init__(self, obs_size, params):
        super().__init__()
        self.rnn_hidden_size = params['rnn_hidden_size']
        self.rnn_layers = params['rnn_layers']

        self.rl_gru = nn.GRU(
            hidden_size=params['rnn_hidden_size'],
            input_size=obs_size,
            num_layers=params['rnn_layers'],
            batch_first=True,
            bias=False,
        )
        self.h_t = torch.zeros(self.rnn_layers, 1, self.rnn_hidden_size)

    def forward(self, x):
        out, _ = self.rl_gru(x)
        return out.squeeze(0).squeeze(0)


class LstmNet(nn.Module):
    """Reference LSTM wrapper (not used by the QuaidSIM-v4 pipeline)."""

    def __init__(self, obs_size, params):
        super().__init__()
        self.rnn_hidden_size = params['rnn_hidden_size']
        self.rnn_layers = params['rnn_layers']
        self.c_t = None

        self.rl_lstm = nn.LSTM(
            hidden_size=params['rnn_hidden_size'],
            input_size=obs_size,
            num_layers=params['rnn_layers'],
            batch_first=True,
        )

    def forward(self, x):
        out, (_, self.c_t) = self.rl_lstm(x.float())
        return out.squeeze(0).squeeze(0)

    def init_state(self):
        pass
