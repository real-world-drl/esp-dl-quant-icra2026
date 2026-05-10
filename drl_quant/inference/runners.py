"""Inference backends — Python ports of the C++ ``Trainer`` subclasses used
by ``Player.cpp``. Each runner abstracts a model-loading + ``select_action``
pair; the player wires them up with a matching preprocessor.

Backends covered (the C++ ``MCUInference`` is intentionally skipped):

* ``TracedRunner`` — TorchScript actor loaded with ``torch.jit.load``. Single
  input, single output. Handles the ``.pt`` / ``.dat`` serialisation the
  upstream training repo emits.
* ``OnnxRunner`` — ONNX actor with a single ``observations`` input. Used for
  non-recurrent networks and for recurrent networks where the GRU lives in
  a separate preprocessor.
* ``OnnxWithRnnRunner`` — ONNX with the GRU baked in (``aug_act_net_*`` /
  ``with_gru_act_net_*``). Two inputs (``observations`` + ``h_t_in``) and two
  outputs (``action`` + ``h_t``); the runner manages ``h_t`` internally.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


class InferenceRunner(ABC):
    """Common interface for a loaded actor."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Run one forward pass. ``state`` is shape (state_dim,) float32;
        returns a shape (action_dim,) float32 array."""

    def reset(self) -> None:
        """Reset any per-episode state (e.g. recurrent hidden state)."""

    def close(self) -> None:
        """Release any backend handles."""


# ---------------------------------------------------------------------------
class TracedRunner(InferenceRunner):
    """TorchScript actor head.

    The C++ training repo emits ``.dat`` files via ``torch::save`` — an
    output archive that carries the parameters but **no forward graph**.
    ``torch.jit.load`` succeeds (you get a ``RecursiveScriptModule``) but
    calling ``module(x)`` / ``module.forward(x)`` raises
    ``AttributeError: 'RecursiveScriptModule' object has no attribute
    'forward'``. The existing ONNX exporters in this repo work around it by
    only using ``scripted.state_dict()``; we apply the same trick here.

    On load we:

    1. Read ``state_dict()`` from the script module.
    2. Detect ``TD3`` vs ``SAC`` from the head key prefix
       (``td3_actor_net.*`` vs ``sac_diag_gaussian_actor.*``) — works
       regardless of filename.
    3. Read the actor's input dim from the first Linear's weight shape and
       the action dim from the last Linear's weight shape.
    4. Construct the matching Python ``TD3Actor`` / ``DiagGaussianActor``
       and transplant the state_dict.

    The rebuilt module is what we call for inference, so we never touch the
    missing forward graph.

    Recurrent variants (``R-`` / ``RA-``) load the same way — the actor
    head input dim is just larger because the GRU output is concatenated
    upstream by the ``GruPreprocessor``.
    """

    def __init__(self, model_path: str, *, device: str = 'cpu') -> None:
        import torch
        from drl_quant.networks.sac import DiagGaussianActor
        from drl_quant.networks.td3 import TD3Actor

        self._torch = torch
        self.device = torch.device(device)
        log.info('loading TorchScript actor: %s (device=%s)', model_path, self.device)

        scripted = torch.jit.load(model_path, map_location=self.device)
        sd = scripted.state_dict()

        # Detect head type from state_dict prefix. State_dict is more
        # robust than filename — handles renamed checkpoints too.
        if any(k.startswith('td3_actor_net.') for k in sd.keys()):
            head_key, algo = 'td3_actor_net', 'TD3'
        elif any(k.startswith('sac_diag_gaussian_actor.') for k in sd.keys()):
            head_key, algo = 'sac_diag_gaussian_actor', 'SAC'
        else:
            raise ValueError(
                f"Could not detect actor type from state_dict in {model_path}. "
                f"Expected 'td3_actor_net.*' or 'sac_diag_gaussian_actor.*' "
                f"prefix; got keys starting with {sorted({k.split('.')[0] for k in sd.keys()})}"
            )

        # First Linear (head_key.0.weight): shape (hidden, input_dim).
        # Last Linear  (head_key.4.weight): shape (action_dim, hidden) for
        # TD3, or (action_dim*2, hidden) for SAC (mu + log_std).
        input_dim = sd[f'{head_key}.0.weight'].shape[1]
        last_out = sd[f'{head_key}.4.weight'].shape[0]
        action_dim = last_out if algo == 'TD3' else last_out // 2

        params = {'hidden_size': 256, 'hidden_size2': 256, 'orthogonal_init': False}
        if algo == 'TD3':
            self._module = TD3Actor(input_dim, action_dim, params)
        else:
            self._module = DiagGaussianActor(input_dim, action_dim, params)

        self._module.load_state_dict(sd)
        self._module.to(self.device).eval()
        log.info('rebuilt %s actor head: input_dim=%d, action_dim=%d',
                 algo, input_dim, action_dim)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0)
            # `module(x)` rather than `module.forward(x)` — the latter is
            # not always exposed as a public attribute on rebuilt modules
            # in newer torch, and `__call__` is the documented entry.
            y = self._module(x).squeeze(0)
        return y.detach().cpu().numpy()


# ---------------------------------------------------------------------------
class OnnxRunner(InferenceRunner):
    """ONNX actor with a single ``observations`` input.

    Use this for non-recurrent actors and for recurrent actors where the GRU
    is run as a separate preprocessor (so the input size already includes
    the GRU's appended features).
    """

    INPUT_NAME = 'observations'
    OUTPUT_NAME = 'action'

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort
        self._ort = ort
        log.info('loading ONNX actor: %s', model_path)
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def select_action(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        out = self._session.run([self._output_name], {self._input_name: x})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
class OnnxWithRnnRunner(InferenceRunner):
    """ONNX with the GRU baked in — ``aug_act_net_*`` / ``with_gru_act_net_*``.

    The graph has two inputs (``observations``, ``h_t_in``) and two outputs
    (``action``, ``h_t``). The runner owns ``h_t`` and resets it to zeros on
    each ``reset()``. Input dimension is the *raw* observation size (17 for
    R-, 25 for RA- since RA- prepends the previous action).
    """

    OBS_INPUT = 'observations'
    HT_INPUT = 'h_t_in'
    ACTION_OUTPUT = 'action'
    HT_OUTPUT = 'h_t'

    def __init__(
        self,
        model_path: str,
        *,
        rnn_layers: int = 3,
        rnn_hidden_size: int = 64,
    ) -> None:
        import onnxruntime as ort
        self._ort = ort
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        log.info('loading ONNX-with-RNN actor: %s (layers=%d, hidden=%d)',
                 model_path, rnn_layers, rnn_hidden_size)
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        # Input/output names follow the export scripts' conventions but we
        # look them up dynamically in case a future export uses different
        # names.
        in_names = [i.name for i in self._session.get_inputs()]
        out_names = [o.name for o in self._session.get_outputs()]
        self._obs_input = self.OBS_INPUT if self.OBS_INPUT in in_names else in_names[0]
        self._ht_input = self.HT_INPUT if self.HT_INPUT in in_names else in_names[1]
        self._action_output = self.ACTION_OUTPUT if self.ACTION_OUTPUT in out_names else out_names[0]
        self._ht_output = self.HT_OUTPUT if self.HT_OUTPUT in out_names else out_names[1]
        self._h_t = self._zero_h()

    def _zero_h(self) -> np.ndarray:
        return np.zeros((self.rnn_layers, 1, self.rnn_hidden_size), dtype=np.float32)

    def reset(self) -> None:
        self._h_t = self._zero_h()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        action, self._h_t = self._session.run(
            [self._action_output, self._ht_output],
            {self._obs_input: x, self._ht_input: self._h_t},
        )
        return np.asarray(action, dtype=np.float32).reshape(-1)
