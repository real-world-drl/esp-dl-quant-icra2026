"""Term-by-term reward tests against hand-computed expected values."""

import math

import numpy as np
import pytest

from quaid_env.config import RewardSettings
from quaid_env.quaid_data import QuaidData
from quaid_env.reward import compute_reward


def _settings():
    s = RewardSettings()
    s.reward_type = 'x-direction'
    s.distance_reward_ratio = 1.0
    s.speed_reward_ratio = 2.0
    s.target_distance_per_step = 0.6
    s.yaw_reward_ratio = 0.5
    s.yaw_delta_reward_ratio = 0.0
    s.pitch_reward_ratio = -0.1
    s.roll_reward_ratio = -0.1
    s.current_reward_ratio = 0.1
    s.current_per_leg_ratio = -0.005
    s.current_cut_off = 4.0
    s.z_reward_ratio = 0.0
    s.z_center = 1.6
    s.acc_z_reward_ratio = -0.025
    s.acc_x_reward_ratio = -0.025
    s.acc_y_reward_ratio = -0.025
    s.action_smoothness_reward_ratio = -0.01
    return s


def _data():
    return QuaidData(
        yaw=0.1, pitch=0.05, roll=-0.05,
        current=2.0,
        current_front_left=2.0, current_front_right=0.5,
        current_back_left=2.5, current_back_right=0.6,
        acc_x=0.2, acc_y=-0.1, acc_z=9.5,
        position_z=1.5,
    )


def test_distance_term_x_direction():
    s = _settings()
    s.speed_reward_ratio = 0  # only test distance
    s.yaw_reward_ratio = 0
    s.pitch_reward_ratio = 0
    s.roll_reward_ratio = 0
    s.current_reward_ratio = 0
    s.current_per_leg_ratio = 0
    s.acc_z_reward_ratio = 0
    s.acc_x_reward_ratio = 0
    s.acc_y_reward_ratio = 0
    s.action_smoothness_reward_ratio = 0
    s.z_reward_ratio = 0
    b = compute_reward(data=_data(), distance=0.4, yaw_delta=0.0,
                       settings=s, action_history=[])
    assert b.distance == pytest.approx(0.4 * 1.0)


def test_distance_term_euclidean_uses_abs():
    s = _settings()
    s.reward_type = 'euclidean'
    b = compute_reward(data=_data(), distance=-0.4, yaw_delta=0.0,
                       settings=s, action_history=[])
    # euclidean takes |distance|.
    assert b.distance == pytest.approx(0.4 * 1.0)


def test_speed_term_peaks_at_target():
    s = _settings()
    b = compute_reward(data=_data(), distance=s.target_distance_per_step,
                       yaw_delta=0.0, settings=s, action_history=[])
    # Gaussian peak: exp(0) * 2.0 = 2.0
    assert b.speed == pytest.approx(2.0)


def test_yaw_term_decays_with_abs_yaw():
    s = _settings()
    d = _data()
    d.yaw = 0.0
    b = compute_reward(data=d, distance=0.0, yaw_delta=0.0,
                       settings=s, action_history=[])
    # exp(0) * 0.5 = 0.5
    assert b.yaw == pytest.approx(0.5)

    d.yaw = 1.0
    b = compute_reward(data=d, distance=0.0, yaw_delta=0.0,
                       settings=s, action_history=[])
    assert b.yaw == pytest.approx(math.exp(-10.0) * 0.5)


def test_per_leg_current_threshold():
    """Only legs with |current| > cutoff/4 contribute. cutoff=4 -> threshold=1."""
    s = _settings()
    d = _data()  # FL=2.0 (above), FR=0.5 (below), BL=2.5 (above), BR=0.6 (below)
    b = compute_reward(data=d, distance=0.0, yaw_delta=0.0,
                       settings=s, action_history=[])
    assert b.current_front_left == pytest.approx(2.0 * -0.005)
    assert b.current_front_right == pytest.approx(0.0)
    assert b.current_back_left == pytest.approx(2.5 * -0.005)
    assert b.current_back_right == pytest.approx(0.0)


def test_action_smoothness_zero_when_history_too_short():
    s = _settings()
    history = [np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32)]
    b = compute_reward(data=_data(), distance=0.0, yaw_delta=0.0,
                       settings=s, action_history=history)
    assert b.action_smoothness == 0.0


def test_action_smoothness_with_three_samples():
    s = _settings()
    a0 = np.zeros(8, dtype=np.float32)
    a1 = np.ones(8, dtype=np.float32)
    a2 = np.full(8, 2.0, dtype=np.float32)
    history = [a0, a1, a2]

    # First-derivative: a2 - a1 = 1
    # Second-derivative: a2 - 2*a1 + a0 = 0
    # sum = 8 * (1 + 0) = 8
    expected = 8.0 * s.action_smoothness_reward_ratio
    b = compute_reward(data=_data(), distance=0.0, yaw_delta=0.0,
                       settings=s, action_history=history)
    assert b.action_smoothness == pytest.approx(expected)


def test_total_is_sum_of_components():
    s = _settings()
    b = compute_reward(data=_data(), distance=0.4, yaw_delta=0.02,
                       settings=s, action_history=[])
    expected = (b.distance + b.speed + b.yaw_delta + b.yaw + b.pitch + b.roll
                + b.z + b.current_front_left + b.current_front_right
                + b.current_back_left + b.current_back_right + b.current
                + b.acc_z + b.acc_x + b.acc_y + b.action_smoothness)
    assert b.total == pytest.approx(expected)
