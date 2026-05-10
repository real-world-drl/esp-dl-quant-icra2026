"""Gymnasium env that ties the MQTT controller, observation builder, and
reward computation together.

This is the Python counterpart to ``QuaidEnv`` in the C++ repo. It does NOT
load any ML model — the inference loop is the consumer's responsibility, so
this env is reusable by any policy framework that speaks the gymnasium API.

Lifecycle::

    settings = load_settings('quaid-icra-sim.yaml')
    env = QuaidEnv(settings)
    env.connect()                        # opens the MQTT session

    obs, info = env.reset()
    for _ in range(settings.robot.max_steps):
        action = policy(obs)             # consumer-supplied; carries h_t etc.
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quaid_env.config import Settings
from quaid_env.mqtt_controller import MqttController
from quaid_env.observations import (
    OBSERVATION_SIZE,
    build_reset_observation,
    build_step_observation,
    yaw_delta_with_wrap,
)
from quaid_env.reward import RewardBreakdown, compute_reward
from quaid_env.theta import ThetaController


log = logging.getLogger(__name__)


ACTION_SIZE = 8


class QuaidEnv(gym.Env):
    """``gymnasium.Env`` for the Quaid quadruped over MQTT."""

    metadata = {'render_modes': []}

    observation_space = spaces.Box(
        low=-2000.0, high=2000.0, shape=(OBSERVATION_SIZE,), dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32,
    )

    def __init__(
        self,
        settings: Settings,
        *,
        controller: Optional[MqttController] = None,
        theta_controller: Optional[ThetaController] = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.controller = controller or MqttController(settings)
        self.theta = theta_controller or ThetaController(settings.robot)

        self._steps = 0
        self._frame = 0
        self._episode_no = 0
        self._mean_yaw = 0.0
        self._last_yaw = 0.0
        self._last_position = (0.0, 0.0)
        self._last_step_ms = int(time.time() * 1000)
        # Deque of recent action vectors (np.float32, shape (8,)) — used by
        # the action-smoothness reward term, which inspects the last 3.
        self._action_history: deque[np.ndarray] = deque(maxlen=3)
        self._connected = False

    # --------------------------------------------------------------- setup
    def connect(self) -> None:
        """Open the MQTT session and push robot-side settings."""
        if self._connected:
            return
        self.controller.connect()
        self.controller.upload_settings()
        self.controller.start_streaming()
        self._connected = True

    # ------------------------------------------------------------- reset --
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if not self._connected:
            self.connect()

        r = self.settings.robot

        # Mirror QuaidEnv.cpp:65-83: pick the reset action based on strategy,
        # then either run a theta adjust or send "pTH0" to reset the rotation.
        strategy = r.reset_strategy
        if strategy == 0:
            time.sleep(r.reset_wait / 1000.0)
        elif strategy == 1:
            self.controller.reset()
            time.sleep(r.reset_wait / 1000.0)
        else:
            self.controller.stand_up()

        if r.adjust_theta_on_reset and r.reset_theta_max_steps > 0:
            self.theta.adjust(self.controller.data, self._mean_yaw, self._frame, self._episode_no, self._steps)
            time.sleep(0.2)
        else:
            self.controller.message('pTH0')

        self._episode_no += 1
        self._steps = 0
        self._action_history.clear()

        data = self.controller.data.snapshot()
        obs = build_reset_observation(data, self.settings.observations)

        self._last_yaw = data.yaw
        self._last_position = (data.position_x, data.position_y)
        self._last_step_ms = int(time.time() * 1000)

        info = {'episode': self._episode_no}
        return obs, info

    # -------------------------------------------------------------- step --
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_SIZE)
        self._steps += 1
        self._frame += 1
        self._action_history.append(action.copy())

        r = self.settings.robot
        terminated = False
        truncated = False

        # Send the action and wait one step period. If we've already hit
        # max_steps the C++ env skips take_step and just sleeps; we mirror
        # that so the env stays in lockstep with the robot.
        if self._steps >= r.max_steps or self.controller.data.snapshot().paused:
            truncated = self._steps >= r.max_steps
            terminated = self.controller.data.snapshot().paused
            time.sleep(r.step_time / 1000.0)
        else:
            self.controller.act(action)
            time.sleep(r.step_time / 1000.0)

        # Read fresh state and derive cross-step quantities.
        now_ms = int(time.time() * 1000)
        time_delta_centisec = (now_ms - self._last_step_ms) / 100.0
        self._last_step_ms = now_ms

        data = self.controller.data.snapshot()
        observation_age_centisec = (now_ms - data.received_time_ms) / 100.0

        if self.settings.reward.reward_type == 'euclidean':
            distance = math.hypot(data.position_x - self._last_position[0],
                                  data.position_y - self._last_position[1])
        else:
            distance = data.position_x - self._last_position[0]

        # Update mean yaw (the C++ formula caps the running average at
        # mean_yaw_steps samples — see QuaidEnv.cpp:181-186).
        denom = min(self._frame, r.mean_yaw_steps) + 1
        self._mean_yaw = (self._mean_yaw * (denom - 1) + abs(data.yaw)) / denom

        if r.yaw_end_episode != 0 and abs(data.yaw) > r.yaw_end_episode:
            log.info('yaw %.3f exceeded limit ±%.3f, terminating episode', data.yaw, r.yaw_end_episode)
            terminated = True

        yaw_delta = yaw_delta_with_wrap(self._last_yaw, data.yaw)

        obs = build_step_observation(
            data,
            self.settings.observations,
            distance=distance,
            yaw_delta=yaw_delta,
            time_delta_centisec=time_delta_centisec,
            observation_age_centisec=observation_age_centisec,
            reward_type=self.settings.reward.reward_type,
        )

        # The C++ env only computes the reward while still stepping (i.e. not
        # in the done-tail), see QuaidEnv.cpp:266-268.
        if self._steps < r.max_steps:
            breakdown = compute_reward(
                data=data,
                distance=distance,
                yaw_delta=yaw_delta,
                settings=self.settings.reward,
                action_history=list(self._action_history),
            )
            reward = breakdown.total
        else:
            breakdown = RewardBreakdown()
            reward = 0.0

        # Mid-episode theta adjust (QuaidEnv.cpp:273-281).
        if r.reset_theta_max_steps > 0 and self._frame >= self.theta.next_reset_frame and self._frame != 0:
            self.theta.adjust(self.controller.data, self._mean_yaw, self._frame, self._episode_no, self._steps)
            self.theta.schedule_next(self._frame)
            time.sleep(0.2)
            data = self.controller.data.snapshot()
            self._last_position = (data.position_x, data.position_y)
            self._last_yaw = data.yaw
        else:
            self._last_position = (data.position_x, data.position_y)
            self._last_yaw = data.yaw

        info = {
            'reward_breakdown': breakdown,
            'mean_yaw': self._mean_yaw,
            'episode': self._episode_no,
            'step': self._steps,
        }
        return obs, float(reward), terminated, truncated, info

    # --------------------------------------------------------------- close
    def close(self) -> None:
        if self._connected:
            self.controller.disconnect()
            self._connected = False
