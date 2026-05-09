"""Inference orchestrator — Python port of ``Player.cpp``.

Given a model file and an env, the Player auto-detects the right runner +
preprocessor, runs ``--episodes`` rollouts, and records timing + reward
statistics. The dispatch logic mirrors ``Player.cpp:27-95``.

Filename heuristics
-------------------

============================================  ========================  =========================
Filename pattern                              Runner                    Preprocessor
============================================  ========================  =========================
``*.pt`` / ``*.dat`` (TorchScript)             ``TracedRunner``          (depends on policy)
``aug_act_net_*.onnx`` /                      ``OnnxWithRnnRunner``     ``NoPreprocessor`` for R-,
``with_gru_act_net_*.onnx`` /                                           ``AddActionsPreprocessor``
``with_rnn_*.onnx``                                                     for RA-
``act_net_*.onnx`` (no aug/with_gru)           ``OnnxRunner``            depends on policy
============================================  ========================  =========================

Policy variants drive the preprocessor when the GRU is *not* baked in:

* ``_TD3_`` / ``_SAC_`` (no ``R-`` / ``RA-``): non-recurrent, ``NoPreprocessor``.
* ``_R-TD3_`` / ``_R-SAC_``: recurrent, external GRU; ``GruPreprocessor`` (or
  ``OnnxGruPreprocessor`` if a ``.onnx`` GRU is supplied).
* ``_RA-TD3_`` / ``_RA-SAC_``: recurrent + previous action;
  ``GruPreprocessor`` with ``actions_to_rnn=True`` for external GRUs, or
  ``AddActionsPreprocessor`` for ``aug_*`` ONNXes (those bake the GRU but
  still need the previous action prepended to the observation).

You can override every choice via ``Player`` constructor arguments.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from drl_quant.inference.preprocessors import (
    AddActionsPreprocessor,
    GruPreprocessor,
    NoPreprocessor,
    OnnxGruPreprocessor,
    Preprocessor,
)
from drl_quant.inference.runners import (
    InferenceRunner,
    OnnxRunner,
    OnnxWithRnnRunner,
    TracedRunner,
)
from drl_quant.inference.stats import EpisodeStats, InferenceStats


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def detect_policy_variant(model_path: str) -> dict:
    """Inspect a model filename and pull out the variant tags the C++ Player
    branches on.

    Returns a dict with::

        {'format': 'onnx' | 'traced',
         'has_gru_inside': bool,        # aug_/with_gru_/with_rnn_
         'is_recurrent': bool,          # R- or RA-
         'actions_to_rnn': bool,        # RA-
         'algorithm': 'TD3' | 'SAC' | None}
    """
    name = Path(model_path).name
    lower = name.lower()

    if name.endswith('.onnx'):
        fmt = 'onnx'
    elif name.endswith('.pt') or name.endswith('.dat'):
        fmt = 'traced'
    else:
        raise ValueError(f'unsupported model extension: {name}')

    has_gru_inside = (
        'aug_act_net' in lower
        or 'aug_' in lower
        or 'with_gru' in lower
        or 'with_rnn' in lower
    )

    is_ra = bool(re.search(r'_RA-', name))
    is_r = bool(re.search(r'_R-', name)) and not is_ra
    is_recurrent = is_ra or is_r

    if 'TD3' in name:
        algo = 'TD3'
    elif 'SAC' in name:
        algo = 'SAC'
    else:
        algo = None

    return {
        'format': fmt,
        'has_gru_inside': has_gru_inside,
        'is_recurrent': is_recurrent,
        'actions_to_rnn': is_ra,
        'algorithm': algo,
    }


# ---------------------------------------------------------------------------
class Player:
    """Run a loaded actor against a gymnasium env and collect statistics.

    Parameters
    ----------
    env: gymnasium.Env
        Already-built env. The QuaidEnv from ``quaid_env`` is the typical
        use case; any env exposing ``reset()`` / ``step(action)`` works.
    model_path: str
        Actor file. ``.onnx``, ``.pt`` or ``.dat``.
    gru_path: str, optional
        Sibling GRU file for recurrent actors that don't bake the GRU in.
        For ``aug_*`` / ``with_gru_*`` ONNX models the GRU is internal so
        this is ignored.
    rnn_layers, rnn_hidden_size: int
        GRU dimensions; defaults match the QuaidSIM-v4 trained policies.
    runner_kind, preprocessor_kind: str, optional
        Force a particular runner / preprocessor instead of auto-detection.
        Valid runners: ``traced``, ``onnx``, ``onnx_with_rnn``.
        Valid preprocessors: ``none``, ``add_actions``, ``gru``, ``onnx_gru``.
    output_dir: str, optional
        If set, writes a SQLite log of inference timings + episode summary.
    test_episodes, max_test_steps, test_step_delay_ms: int
        Loop bounds + per-step throttle. Most users want the env's own
        ``step_time`` to gate the rate, so default delay is 0.
    device: str
        Torch device for TorchScript paths.
    """

    def __init__(
        self,
        env,
        model_path: str,
        *,
        gru_path: Optional[str] = None,
        rnn_layers: int = 3,
        rnn_hidden_size: int = 64,
        runner_kind: Optional[str] = None,
        preprocessor_kind: Optional[str] = None,
        output_dir: Optional[str] = None,
        test_episodes: int = 5,
        max_test_steps: int = 500,
        test_step_delay_ms: int = 0,
        device: str = 'cpu',
    ) -> None:
        self.env = env
        self.model_path = model_path
        self.gru_path = gru_path
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.test_episodes = test_episodes
        self.max_test_steps = max_test_steps
        self.test_step_delay_ms = test_step_delay_ms
        self.device = device

        self._action_dim = int(np.prod(env.action_space.shape))
        self._obs_dim = int(np.prod(env.observation_space.shape))

        info = detect_policy_variant(model_path)
        log.info('detected: %s', info)
        self._info = info

        runner_kind = runner_kind or self._auto_runner_kind(info)
        preprocessor_kind = preprocessor_kind or self._auto_preprocessor_kind(info)
        log.info('runner=%s, preprocessor=%s', runner_kind, preprocessor_kind)

        self.runner: InferenceRunner = self._build_runner(runner_kind)
        self.preprocessor: Preprocessor = self._build_preprocessor(preprocessor_kind, info)
        self.stats = InferenceStats(output_dir=output_dir)

    # ---------------------------------------------------- factories ---
    @staticmethod
    def _auto_runner_kind(info: dict) -> str:
        if info['format'] == 'traced':
            return 'traced'
        return 'onnx_with_rnn' if info['has_gru_inside'] else 'onnx'

    @staticmethod
    def _auto_preprocessor_kind(info: dict) -> str:
        if info['has_gru_inside']:
            # GRU is inside the model. RA- still needs prev_action prepended.
            return 'add_actions' if info['actions_to_rnn'] else 'none'
        if info['is_recurrent']:
            # Recurrent actor head with a separate GRU somewhere.
            return 'gru'
        return 'none'

    def _build_runner(self, kind: str) -> InferenceRunner:
        if kind == 'traced':
            return TracedRunner(self.model_path, device=self.device)
        if kind == 'onnx':
            return OnnxRunner(self.model_path)
        if kind == 'onnx_with_rnn':
            return OnnxWithRnnRunner(
                self.model_path,
                rnn_layers=self.rnn_layers,
                rnn_hidden_size=self.rnn_hidden_size,
            )
        raise ValueError(f'unknown runner kind: {kind}')

    def _build_preprocessor(self, kind: str, info: dict) -> Preprocessor:
        if kind == 'none':
            return NoPreprocessor()
        if kind == 'add_actions':
            return AddActionsPreprocessor(action_dim=self._action_dim)
        if kind == 'gru':
            if not self.gru_path:
                raise ValueError('GruPreprocessor requested but --gru-path not supplied')
            rnn_input_size = self._obs_dim + (self._action_dim if info['actions_to_rnn'] else 0)
            if self.gru_path.endswith('.onnx'):
                return OnnxGruPreprocessor(
                    self.gru_path,
                    rnn_hidden_size=self.rnn_hidden_size,
                    rnn_layers=self.rnn_layers,
                    actions_to_rnn=info['actions_to_rnn'],
                    action_dim=self._action_dim,
                )
            return GruPreprocessor(
                self.gru_path,
                rnn_input_size=rnn_input_size,
                rnn_hidden_size=self.rnn_hidden_size,
                rnn_layers=self.rnn_layers,
                actions_to_rnn=info['actions_to_rnn'],
                action_dim=self._action_dim,
                device=self.device,
            )
        if kind == 'onnx_gru':
            if not self.gru_path:
                raise ValueError('OnnxGruPreprocessor requested but --gru-path not supplied')
            return OnnxGruPreprocessor(
                self.gru_path,
                rnn_hidden_size=self.rnn_hidden_size,
                rnn_layers=self.rnn_layers,
                actions_to_rnn=info['actions_to_rnn'],
                action_dim=self._action_dim,
            )
        raise ValueError(f'unknown preprocessor kind: {kind}')

    # ---------------------------------------------------- play loop ---
    def play(self) -> InferenceStats:
        """Run ``test_episodes`` rollouts and return the populated stats
        object. The C++ Player prints the per-episode table at the end; we
        do the same after the loop unless ``stats.print_summary`` is
        suppressed by the caller.
        """
        # Extra reset before the loop matches the C++ behaviour
        # (Trainer::check_progress: "extra reset - seems to be necessary").
        obs, _ = self.env.reset()
        self.preprocessor.reset()
        self.runner.reset()
        time.sleep(1.0)

        prev_action = np.zeros(self._action_dim, dtype=np.float32)

        for episode_no in range(self.test_episodes):
            obs, _ = self.env.reset()
            self.preprocessor.reset()
            self.runner.reset()
            prev_action = np.zeros(self._action_dim, dtype=np.float32)

            episode_reward = 0.0
            episode_start = time.perf_counter()
            inference_times: list[int] = []

            self._wait_if_paused()

            for step in range(self.max_test_steps):
                inference_start = time.perf_counter_ns()
                state = self.preprocessor.process(obs, prev_action)
                action = self.runner.select_action(state)
                inference_times.append((time.perf_counter_ns() - inference_start) // 1000)

                action = np.clip(np.asarray(action, dtype=np.float32),
                                 self.env.action_space.low, self.env.action_space.high)

                obs, reward, terminated, truncated, _info = self.env.step(action)
                episode_reward += float(reward)
                prev_action = action.copy()

                if terminated or truncated:
                    break

                if self.test_step_delay_ms > 0:
                    time.sleep(self.test_step_delay_ms / 1000.0)
            else:
                step = self.max_test_steps - 1

            wall = time.perf_counter() - episode_start
            ep = EpisodeStats(
                episode_no=episode_no,
                reward=episode_reward,
                steps=step + 1,
                wall_seconds=wall,
                inference_times_us=inference_times,
            )
            self.stats.record_episode(ep)
            log.info('episode %d: reward=%.3f steps=%d fps=%.2f mean_inf_us=%.1f',
                     episode_no + 1, ep.reward, ep.steps, ep.fps, ep.mean_inference_us)

            self._wait_if_paused()

        return self.stats

    # ---------------------------------------------------- pause -------
    def _wait_if_paused(self) -> None:
        """Mirror ``Trainer::pause_test`` — block while the env reports a
        pause signal. The QuaidEnv exposes ``controller.data.snapshot().paused``;
        we look for a ``paused`` attribute and fall back to a no-op."""
        env = self.env
        # Check via attribute path env.controller.data.snapshot().paused
        # (QuaidEnv) or env.paused (custom envs). No-op if neither exists.
        for _ in range(60):
            paused = False
            try:
                paused = bool(env.controller.data.snapshot().paused)
            except AttributeError:
                paused = bool(getattr(env, 'paused', False))
            if not paused:
                return
            log.info('env paused — waiting...')
            time.sleep(2.0)

    def close(self) -> None:
        self.runner.close()
        self.stats.close()
