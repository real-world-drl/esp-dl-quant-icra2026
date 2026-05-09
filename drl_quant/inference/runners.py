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

    The C++ training repo emits ``.dat`` files via ``torch::jit::save``; these
    are the same TorchScript serialisation Python uses for ``.pt`` and load
    via ``torch.jit.load``. Recurrent variants (``R-`` / ``RA-``) use
    ``act_net_*`` files that contain only the actor head — the GRU lives in
    a sibling ``rnn_*`` file consumed by ``GruPreprocessor``.
    """

    def __init__(self, model_path: str, *, device: str = 'cpu') -> None:
        import torch
        self._torch = torch
        self.device = torch.device(device)
        log.info('loading TorchScript actor: %s (device=%s)', model_path, self.device)
        self._module = torch.jit.load(model_path, map_location=self.device)
        self._module.eval()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0)
            y = self._module.forward(x).squeeze(0)
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
