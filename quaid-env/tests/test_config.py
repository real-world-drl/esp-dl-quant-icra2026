"""Config loader tests — make sure the OpenCV %YAML:1.0 prefix is handled
and all the keys from the bundled examples are populated."""

import math
from pathlib import Path

import pytest

from quaid_env.config import load


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'


def test_load_quaid_icra_sim_yaml():
    s = load(EXAMPLES / 'quaid-icra-sim.yaml')

    assert s.ports.mqtt_server_ip == 'tcp://mqtt-server:1883'
    # mqtt_queue_no is an int in the YAML but the loader coerces to str so
    # it can be slotted into MQTT topic format strings unchanged. Don't
    # pin to a specific value — users routinely edit it for their own
    # broker / queue layout.
    assert isinstance(s.ports.mqtt_queue_no, str)
    assert s.ports.mqtt_queue_no.isdigit()

    # Robot section.
    assert s.robot.max_steps == 500
    assert s.robot.step_time == 50
    assert s.robot.streaming_delay == 25
    assert s.robot.exponential_filter == 0.0
    assert s.robot.adjust_theta_distribution == 'zero'
    # PI/8 in the YAML.
    assert s.robot.adjust_theta_interval == pytest.approx(0.392699081)
    # PI/2.
    assert s.robot.yaw_end_episode == pytest.approx(1.570796327)

    # Observations: per-leg + acc_z + yaw + pitch + roll on; the rest off.
    assert s.observations.distance is False
    assert s.observations.voltage is False
    assert s.observations.time_delta is False
    assert s.observations.observation_age is False
    assert s.observations.yaw_delta is False
    assert s.observations.current is True
    assert s.observations.current_per_leg is True
    assert s.observations.acc_z is True
    assert s.observations.yaw is True
    assert s.observations.pitch is True
    assert s.observations.roll is True
    assert s.observations.normalization_type == 'divide_by_max'

    # Reward section — checking key ratios from the YAML.
    assert s.reward.reward_type == 'x-direction'
    assert s.reward.distance_reward_ratio == pytest.approx(1.0)
    assert s.reward.speed_reward_ratio == pytest.approx(2.0)
    assert s.reward.target_distance_per_step == pytest.approx(0.6)
    assert s.reward.yaw_reward_ratio == pytest.approx(0.5)
    assert s.reward.pitch_reward_ratio == pytest.approx(-0.1)
    assert s.reward.current_reward_ratio == pytest.approx(0.1)
    assert s.reward.current_per_leg_ratio == pytest.approx(-0.005)
    assert s.reward.action_smoothness_reward_ratio == pytest.approx(-0.01)
    assert s.reward.acc_z_reward_ratio == pytest.approx(-0.025)

    # current_cut_off is mirrored from observations to reward.
    assert s.reward.current_cut_off == s.observations.current_cut_off == pytest.approx(4.0)


def test_real_yaml_has_sim_zero():
    s = load(EXAMPLES / 'quaid-icra-real.yaml')
    # The only meaningful difference between -sim and -real is `robot.sim`.
    assert s.robot.sim is False
