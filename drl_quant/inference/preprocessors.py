"""Observation preprocessors — Python ports of the C++ ``ObservationPreprocessor``
hierarchy used by ``Player.cpp``.

The preprocessor is responsible for shaping the raw env observation into
whatever the actor expects. Combined with the inference runners
(``runners.py``) you get the four C++ deployment paths:

============================  ===================  =========================
Combo                          Runner               Preprocessor
============================  ===================  =========================
Non-recurrent TD3 / SAC        ``OnnxRunner`` /
                               ``TracedRunner``     ``NoPreprocessor``
ONNX with Aug-GRU baked in     ``OnnxWithRnnRunner`` ``NoPreprocessor`` (R-)
                                                    ``AddActionsPreprocessor`` (RA-)
ONNX without GRU + ext GRU     ``OnnxRunner``        ``GruPreprocessor`` /
                                                    ``OnnxGruPreprocessor``
TorchScript actor + ext GRU    ``TracedRunner``      ``GruPreprocessor`` /
                                                    ``OnnxGruPreprocessor``
============================  ===================  =========================

The "ONNX without GRU" paths exist for deployments where the GRU runs as a
separate model — useful for benchmarking quantised actor heads against an
unquantised GRU. The bundled QuaidSIM-v4 ONNXes all bake the GRU in.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


class Preprocessor(ABC):
    """Builds the actor's input from (observation, previous_action)."""

    @abstractmethod
    def process(self, observation: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        ...

    def reset(self) -> None:
        """Reset any per-episode state (e.g. recurrent hidden state)."""

    @property
    def output_size_extra(self) -> int:
        """How many dims this preprocessor adds on top of the raw observation
        (used by the player to compute the actor's expected input size)."""
        return 0


# ---------------------------------------------------------------------------
class NoPreprocessor(Preprocessor):
    """Pass-through. Used for non-recurrent actors and for ``aug_*`` ONNX of
    the R- variant where the actor takes the raw 17-dim observation."""

    def process(self, observation: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)


# ---------------------------------------------------------------------------
class AddActionsPreprocessor(Preprocessor):
    """Prepend the previous action vector to the observation.

    Mirrors ``AddActionsPreprocessor.cpp``: ``cat(action, observation)``.
    Used for RA- variants when the actor expects ``[prev_action;
    observation]`` as a single tensor — e.g. ``aug_act_net_*_RA-*.onnx``.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim

    @property
    def output_size_extra(self) -> int:
        return self.action_dim

    def process(self, observation: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        if prev_action is None or len(prev_action) == 0:
            prev_action = np.zeros(self.action_dim, dtype=np.float32)
        return np.concatenate(
            [np.asarray(prev_action, dtype=np.float32),
             np.asarray(observation, dtype=np.float32)],
        )


# ---------------------------------------------------------------------------
class GruPreprocessor(Preprocessor):
    """Run a separate TorchScript GRU on the observation (or
    ``[observation; prev_action]`` for ``actions_to_rnn=True``) and return
    ``cat(rnn_input, gru_features)``.

    Mirrors ``GruPreprocessor.cpp`` semantics. The output size is
    ``input_size + rnn_hidden_size`` where ``input_size`` is the raw
    observation (plus the action dim if ``actions_to_rnn``).

    This preprocessor is only needed when the GRU is *not* baked into the
    actor — i.e. you have a non-recurrent ``act_net_*.onnx`` + a separate
    ``rnn_*.dat`` and want to compose them at inference time.
    """

    def __init__(
        self,
        gru_path: str,
        *,
        rnn_input_size: int,
        rnn_hidden_size: int = 64,
        rnn_layers: int = 3,
        actions_to_rnn: bool = False,
        action_dim: int = 0,
        device: str = 'cpu',
    ) -> None:
        import torch
        from drl_quant.networks.rnn import GruNet
        self._torch = torch
        self.device = torch.device(device)
        self.actions_to_rnn = actions_to_rnn
        self.action_dim = action_dim
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        log.info('loading TorchScript GRU: %s (input=%d, hidden=%d, layers=%d, actions_to_rnn=%s)',
                 gru_path, rnn_input_size, rnn_hidden_size, rnn_layers, actions_to_rnn)
        # Load weights from the upstream TorchScript file via state_dict
        # transfer (rather than calling the TorchScript module directly) so
        # the forward signature is fixed regardless of how the upstream
        # module was scripted.
        scripted = torch.jit.load(gru_path, map_location=self.device)
        params = {'rnn_hidden_size': rnn_hidden_size, 'rnn_layers': rnn_layers}
        self._gru = GruNet(rnn_input_size, params).to(self.device).eval()
        self._gru.load_state_dict(scripted.state_dict())
        self._gru.rl_gru.flatten_parameters()
        self._h_t: Optional['torch.Tensor'] = None
        self.reset()

    @property
    def output_size_extra(self) -> int:
        # We append rnn_hidden_size GRU features. The action_dim adds happen
        # via the preprocessor's input shape, not its output extra.
        return self.rnn_hidden_size + (self.action_dim if self.actions_to_rnn else 0)

    def reset(self) -> None:
        torch = self._torch
        self._h_t = torch.zeros(self.rnn_layers, 1, self.rnn_hidden_size, device=self.device)

    def process(self, observation: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(observation, dtype=np.float32)).to(self.device)
            if self.actions_to_rnn:
                if prev_action is None or len(prev_action) == 0:
                    prev_action = np.zeros(self.action_dim, dtype=np.float32)
                act_t = torch.from_numpy(np.asarray(prev_action, dtype=np.float32)).to(self.device)
                rnn_input = torch.cat([obs_t, act_t]).unsqueeze(0).unsqueeze(0)
            else:
                rnn_input = obs_t.unsqueeze(0).unsqueeze(0)

            features, self._h_t = self._gru.forward(rnn_input, self._h_t)
            stacked = torch.cat([rnn_input.squeeze(0).squeeze(0), features.reshape(-1)])
        return stacked.detach().cpu().numpy()


# ---------------------------------------------------------------------------
class OnnxGruPreprocessor(Preprocessor):
    """Run a separate ONNX GRU on the observation and return
    ``cat(rnn_input, gru_features)``.

    Mirrors ``OnnxGruPreprocessor.cpp``. The ONNX GRU graph must have inputs
    ``[observations, h_t_in]`` and outputs ``[features, h_t]`` — matching
    what ``drl_quant.onnx_export.onnx_export_gru`` produces.
    """

    def __init__(
        self,
        gru_path: str,
        *,
        rnn_hidden_size: int = 64,
        rnn_layers: int = 3,
        actions_to_rnn: bool = False,
        action_dim: int = 0,
    ) -> None:
        import onnxruntime as ort
        self._ort = ort
        self.actions_to_rnn = actions_to_rnn
        self.action_dim = action_dim
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        log.info('loading ONNX GRU: %s', gru_path)
        self._session = ort.InferenceSession(gru_path, providers=['CPUExecutionProvider'])
        in_names = [i.name for i in self._session.get_inputs()]
        out_names = [o.name for o in self._session.get_outputs()]
        self._obs_input = in_names[0]
        self._ht_input = in_names[1]
        self._features_output = out_names[0]
        self._ht_output = out_names[1]
        self._h_t = self._zero_h()

    def _zero_h(self) -> np.ndarray:
        return np.zeros((self.rnn_layers, 1, self.rnn_hidden_size), dtype=np.float32)

    @property
    def output_size_extra(self) -> int:
        return self.rnn_hidden_size + (self.action_dim if self.actions_to_rnn else 0)

    def reset(self) -> None:
        self._h_t = self._zero_h()

    def process(self, observation: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        rnn_input = np.asarray(observation, dtype=np.float32)
        if self.actions_to_rnn:
            if prev_action is None or len(prev_action) == 0:
                prev_action = np.zeros(self.action_dim, dtype=np.float32)
            rnn_input = np.concatenate([rnn_input, np.asarray(prev_action, dtype=np.float32)])

        features, self._h_t = self._session.run(
            [self._features_output, self._ht_output],
            {self._obs_input: rnn_input.reshape(1, -1), self._ht_input: self._h_t},
        )
        return np.concatenate([rnn_input, np.asarray(features, dtype=np.float32).reshape(-1)])
