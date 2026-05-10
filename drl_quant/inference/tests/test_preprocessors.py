"""Preprocessor tests — specifically the **input order**, which is
load-bearing because the trained actor heads were exposed to a specific
``[action | observation]`` layout in the C++ pipeline.

If any of these tests fail, the most likely reason is someone reversed
``np.concatenate`` arguments thinking the order was arbitrary. It is not.
The robot will still get actions out the other end — they will just be
nonsense and the robot won't move.

Reference: ``sim-to-real-cpp/src/training/GruPreprocessor.cpp:39-44`` and
``AddActionsPreprocessor.cpp:14-19``.
"""

import numpy as np
import pytest

from drl_quant.inference.preprocessors import (
    AddActionsPreprocessor,
    GruPreprocessor,
)


# -------------------------------------------- AddActionsPreprocessor ----
def test_add_actions_preprocessor_action_first():
    """``cat([action, observation])``. Locked in by the C++ deployment."""
    pp = AddActionsPreprocessor(action_dim=8)
    obs = np.arange(17, dtype=np.float32) + 100.0   # 100..116
    act = np.arange(8, dtype=np.float32)            # 0..7
    out = pp.process(obs, act)

    assert out.shape == (25,)
    # First 8 dims must be the action.
    np.testing.assert_array_equal(out[:8], act)
    # Next 17 dims must be the observation.
    np.testing.assert_array_equal(out[8:], obs)


def test_add_actions_preprocessor_handles_missing_prev_action():
    pp = AddActionsPreprocessor(action_dim=8)
    obs = np.full(17, 0.5, dtype=np.float32)
    out = pp.process(obs, np.array([], dtype=np.float32))

    assert out.shape == (25,)
    # Missing action -> zero-filled.
    np.testing.assert_array_equal(out[:8], np.zeros(8))


# ---------------------------------------------------- GruPreprocessor ----
class _FakeGru:
    """Drop-in replacement for the loaded TorchScript GRU — lets us
    exercise GruPreprocessor without needing a .dat on disk."""

    def __init__(self, hidden_size, layers):
        self.hidden_size = hidden_size
        self.layers = layers

    def forward(self, x, h_t):
        # Shape contract: x is (1, 1, input_dim); we return features of
        # shape (1, hidden_size) and a new h_t identical to the input
        # (deterministic, easy to assert against).
        import torch
        features = torch.zeros(1, self.hidden_size)
        return features, h_t

    def to(self, *a, **kw):
        return self

    def eval(self):
        return self


def _gru_with_fake(action_dim, hidden_size=64, layers=3, actions_to_rnn=True):
    """Build a GruPreprocessor and stub out the loaded GRU so the test
    doesn't need a .dat file. We only assert on the layout of `rnn_input`
    that the preprocessor produces, not the GRU's transformation."""
    pp = GruPreprocessor.__new__(GruPreprocessor)
    import torch
    pp._torch = torch
    pp.device = torch.device('cpu')
    pp.actions_to_rnn = actions_to_rnn
    pp.action_dim = action_dim
    pp.rnn_layers = layers
    pp.rnn_hidden_size = hidden_size
    pp._gru = _FakeGru(hidden_size, layers)
    pp._h_t = None
    pp.reset()
    return pp


def test_gru_preprocessor_action_first_then_obs():
    """``rnn_input = cat([action, observation])`` — load-bearing.
    Output dims: first ``action_dim`` are the action, next ``obs_dim`` are
    the observation, last ``hidden_size`` are GRU features."""
    pp = _gru_with_fake(action_dim=8, hidden_size=64, actions_to_rnn=True)
    obs = np.arange(17, dtype=np.float32) + 100.0   # 100..116
    act = np.arange(8, dtype=np.float32)            # 0..7
    out = pp.process(obs, act)

    assert out.shape == (8 + 17 + 64,)              # 89 for RA-
    # First 8 dims: action.
    np.testing.assert_array_equal(out[:8], act)
    # Next 17 dims: observation.
    np.testing.assert_array_equal(out[8:25], obs)
    # Last 64 dims: GRU features (zeros from _FakeGru).
    np.testing.assert_array_equal(out[25:], np.zeros(64))


def test_gru_preprocessor_no_actions_passes_obs_only():
    """``actions_to_rnn=False`` (R- variants): rnn_input is just the
    observation. Output is then ``[obs | gru_features]``."""
    pp = _gru_with_fake(action_dim=8, hidden_size=64, actions_to_rnn=False)
    obs = np.arange(17, dtype=np.float32) + 100.0
    out = pp.process(obs, np.zeros(8, dtype=np.float32))

    assert out.shape == (17 + 64,)                  # 81 for R-
    np.testing.assert_array_equal(out[:17], obs)
    np.testing.assert_array_equal(out[17:], np.zeros(64))


def test_gru_preprocessor_handles_missing_prev_action():
    pp = _gru_with_fake(action_dim=8, hidden_size=64, actions_to_rnn=True)
    obs = np.full(17, 0.5, dtype=np.float32)
    out = pp.process(obs, np.array([], dtype=np.float32))

    assert out.shape == (89,)
    # First 8 dims should be the implicit zero action.
    np.testing.assert_array_equal(out[:8], np.zeros(8))
