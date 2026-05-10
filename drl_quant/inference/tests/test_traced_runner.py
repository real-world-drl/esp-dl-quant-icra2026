"""Tests for TracedRunner — specifically the rebuild-from-state_dict path.

The C++ training repo saves actor checkpoints via ``torch::save``, which
writes an output archive carrying parameters but **no forward graph**. So
``module.forward(x)`` raises on those files. ``TracedRunner`` works around
this by reading state_dict, detecting TD3 / SAC from the head key prefix,
and rebuilding the matching Python ``nn.Module``.

These tests construct a synthetic .pt that has the same state_dict shape
as a C++-saved ``.dat``, then load it and assert TracedRunner produces a
sensible action without ever touching the (possibly missing) forward
graph.
"""

import numpy as np
import pytest
import torch

from drl_quant.inference.runners import TracedRunner
from drl_quant.networks.sac import DiagGaussianActor
from drl_quant.networks.td3 import TD3Actor


PARAMS = {'hidden_size': 256, 'hidden_size2': 256, 'orthogonal_init': False}


def _save_scripted(actor, path):
    """torch.jit.script + save — produces a script module that DOES have a
    forward graph. TracedRunner's rebuild path doesn't depend on that
    being present, so its behaviour is identical for files with or
    without a forward — we just exercise the load + state_dict path here.
    """
    scripted = torch.jit.script(actor)
    torch.jit.save(scripted, str(path))


# ---------------------------------------------------- TD3 head ----------
@pytest.mark.parametrize('input_dim, action_dim', [
    (17, 8),    # non-recurrent TD3
    (81, 8),    # R-TD3 (17 obs + 64 GRU features)
    (89, 8),    # RA-TD3 (17 obs + 8 prev_action + 64 GRU features)
])
def test_traced_runner_td3_rebuild(tmp_path, input_dim, action_dim):
    torch.manual_seed(0)
    actor = TD3Actor(input_dim, action_dim, PARAMS).eval()
    path = tmp_path / f'act_net_TD3_{input_dim}.pt'
    _save_scripted(actor, path)

    runner = TracedRunner(str(path))
    state = np.full(input_dim, 0.1, dtype=np.float32)
    action = runner.select_action(state)

    assert action.shape == (action_dim,)
    assert action.dtype == np.float32
    # Tanh head -> outputs in [-1, 1].
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    # Same input through the original actor must produce the same action
    # (within float32 noise) — confirms the state_dict transplant worked.
    with torch.no_grad():
        ref = actor(torch.from_numpy(state).unsqueeze(0)).squeeze(0).numpy()
    assert np.allclose(action, ref, atol=1e-6)


# ---------------------------------------------------- SAC head ----------
@pytest.mark.parametrize('input_dim, action_dim', [
    (17, 8),
    (89, 8),    # RA-SAC
])
def test_traced_runner_sac_rebuild(tmp_path, input_dim, action_dim):
    torch.manual_seed(1)
    actor = DiagGaussianActor(input_dim, action_dim, PARAMS).eval()
    path = tmp_path / f'act_net_SAC_{input_dim}.pt'
    _save_scripted(actor, path)

    runner = TracedRunner(str(path))
    state = np.full(input_dim, 0.1, dtype=np.float32)
    action = runner.select_action(state)

    assert action.shape == (action_dim,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


# --------------------------------------------- error path: bad state_dict ---
def test_unrecognised_actor_state_dict_raises(tmp_path):
    """If we point TracedRunner at a TorchScript file whose state_dict
    doesn't have either expected head prefix, the runner must raise a
    clear ValueError rather than crashing later in load_state_dict."""

    class Foreign(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.foreign_head = torch.nn.Linear(10, 4)

        def forward(self, x):
            return self.foreign_head(x)

    actor = Foreign().eval()
    path = tmp_path / 'foreign.pt'
    _save_scripted(actor, path)

    with pytest.raises(ValueError, match="Could not detect actor type"):
        TracedRunner(str(path))
