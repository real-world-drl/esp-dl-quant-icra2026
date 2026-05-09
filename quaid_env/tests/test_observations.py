"""Slot-layout and normalization tests for the observation builder.

These tests are the policy-compatibility safety net: the trained QuaidSIM-v4
policies were trained against the slot order produced by ``QuaidEnv`` in
the C++ project, and any deviation here will silently degrade their
behaviour at deploy time.
"""

import math

import numpy as np
import pytest

from quaid_env.config import ObservationSettings
from quaid_env.observations import (
    IND_ACC_Z,
    IND_CURRENT,
    IND_CURRENT_BACK_LEFT,
    IND_CURRENT_BACK_RIGHT,
    IND_CURRENT_FRONT_LEFT,
    IND_CURRENT_FRONT_RIGHT,
    IND_DISTANCE,
    IND_OBS_AGE,
    IND_PITCH,
    IND_ROLL,
    IND_SERVO_0,
    IND_TIME_DELTA,
    IND_VOLTAGE,
    IND_YAW,
    IND_YAW_DELTA,
    OBSERVATION_SIZE,
    build_step_observation,
    yaw_delta_with_wrap,
)
from quaid_env.quaid_data import QuaidData


def _quaid_v4_settings():
    """Settings that match config/quaid-icra-sim.yaml."""
    s = ObservationSettings()
    s.distance = False
    s.voltage = False
    s.time_delta = False
    s.observation_age = False
    s.yaw_delta = False
    s.current = True
    s.current_per_leg = True
    s.yaw = True
    s.pitch = True
    s.roll = True
    s.acc_z = True
    s.normalization_type = 'divide_by_max'
    return s


def _quaid_data_with_currents():
    return QuaidData(
        yaw=0.1, pitch=0.05, roll=-0.05,
        current=2.0,
        current_front_left=0.5, current_front_right=0.6,
        current_back_left=0.7, current_back_right=0.8,
        acc_z=9.8,
        position_knee_back_left=100, position_thigh_back_left=110,
        position_knee_back_right=120, position_thigh_back_right=130,
        position_knee_front_right=140, position_thigh_front_right=150,
        position_knee_front_left=160, position_thigh_front_left=170,
    )


def test_observation_shape_and_dtype():
    obs = build_step_observation(
        _quaid_data_with_currents(), _quaid_v4_settings(),
        distance=0.0, yaw_delta=0.0,
        time_delta_centisec=0.0, observation_age_centisec=0.0,
        reward_type='x-direction',
    )
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32


def test_quaidsim_v4_layout_per_leg_currents_at_aliased_slots():
    """With per_leg + acc_z enabled and time_delta/distance/voltage/obs_age
    disabled, slots 0/1/2/8 should hold the four leg currents and slot 5
    should hold acc_z (matches QuaidEnv.cpp:255-263)."""
    s = _quaid_v4_settings()
    d = _quaid_data_with_currents()

    obs = build_step_observation(
        d, s,
        distance=99.0,        # ignored (settings.distance is False)
        yaw_delta=99.0,       # ignored (settings.yaw_delta is False)
        time_delta_centisec=99.0,
        observation_age_centisec=99.0,
        reward_type='x-direction',
    )

    # Per-leg currents at the aliased slots.
    assert obs[IND_CURRENT_FRONT_LEFT] == pytest.approx(d.current_front_left / 4.0)
    assert obs[IND_CURRENT_FRONT_RIGHT] == pytest.approx(d.current_front_right / 4.0)
    assert obs[IND_CURRENT_BACK_LEFT] == pytest.approx(d.current_back_left / 4.0)
    assert obs[IND_CURRENT_BACK_RIGHT] == pytest.approx(d.current_back_right / 4.0)

    # Total current at slot 3.
    assert obs[IND_CURRENT] == pytest.approx(d.current / 8.0)

    # Yaw / pitch / roll normalised by π/2.
    half_pi = math.pi / 2
    assert obs[IND_YAW] == pytest.approx(d.yaw / half_pi)
    assert obs[IND_PITCH] == pytest.approx(d.pitch / half_pi)
    assert obs[IND_ROLL] == pytest.approx(d.roll / half_pi)

    # Acc_z overrides yaw_delta slot.
    assert obs[IND_ACC_Z] == pytest.approx(d.acc_z / 20.0)

    # Servos in order: BL knee, BL thigh, BR knee, BR thigh,
    #                  FR knee, FR thigh, FL knee, FL thigh.
    expected_servos = [
        d.position_knee_back_left, d.position_thigh_back_left,
        d.position_knee_back_right, d.position_thigh_back_right,
        d.position_knee_front_right, d.position_thigh_front_right,
        d.position_knee_front_left, d.position_thigh_front_left,
    ]
    for i, exp in enumerate(expected_servos):
        assert obs[IND_SERVO_0 + i] == pytest.approx(exp / 500.0)


def test_layout_with_primary_fields_enabled():
    """Flip the gates: enable time_delta/distance/voltage/observation_age,
    disable current_per_leg. Slots 0/1/2/8 should now hold those primaries
    in their normalized form, NOT the per-leg currents."""
    s = _quaid_v4_settings()
    s.distance = True
    s.voltage = True
    s.time_delta = True
    s.observation_age = True
    s.current_per_leg = False
    s.acc_z = False
    s.yaw_delta = True

    d = _quaid_data_with_currents()
    d.voltage = 1500.0

    obs = build_step_observation(
        d, s,
        distance=0.5, yaw_delta=0.02,
        time_delta_centisec=5.0, observation_age_centisec=3.0,
        reward_type='x-direction',
    )

    assert obs[IND_TIME_DELTA] == pytest.approx(5.0)        # raw, no normalisation
    assert obs[IND_DISTANCE] == pytest.approx(0.5 / 2.0)    # /max=2
    assert obs[IND_VOLTAGE] == pytest.approx(1500.0 / 2500.0)
    assert obs[IND_OBS_AGE] == pytest.approx(3.0)
    assert obs[IND_YAW_DELTA] == pytest.approx(0.02 / 1.0)

    # Per-leg currents must NOT have leaked in.
    assert obs[IND_CURRENT_FRONT_LEFT] == pytest.approx(5.0)  # = time_delta slot
    assert obs[IND_CURRENT_FRONT_RIGHT] == pytest.approx(0.5 / 2.0)


def test_yaw_delta_wraps_around_pi():
    # Crossing from +3.0 to -3.0 should produce the short-way delta (≈0.283),
    # not the raw |a-b|=6.0. |current|==|last| so no negation.
    delta = yaw_delta_with_wrap(last_yaw=3.0, current_yaw=-3.0)
    assert delta == pytest.approx(2 * math.pi - 6.0)


def test_yaw_delta_sign_is_negative_when_drifting_away_from_zero():
    # From 0.1 to 0.3: |current|>|last| so the env penalises with negative.
    delta = yaw_delta_with_wrap(last_yaw=0.1, current_yaw=0.3)
    assert delta == pytest.approx(-0.2)


def test_yaw_delta_sign_is_positive_when_returning_toward_zero():
    delta = yaw_delta_with_wrap(last_yaw=0.3, current_yaw=0.1)
    assert delta == pytest.approx(0.2)
