"""Unit tests for the MqttController's packet decode path.

The controller's network plumbing (paho-mqtt connection, threading) is hard
to unit-test, but the message-handler functions are pure: they take a byte
payload and mutate ``SharedQuaidData``. Drive them directly so the tests
don't need a broker.

The most important thing locked in here is the **mocap unit**: the
simulator publishes positions in decimetres (0.1 m) and the controller has
to keep that unit unchanged so the reward YAML's
``target_distance_per_step`` / ``z_center`` (also in dm) match. See
``mqtt_controller._handle_mocap`` for the load-bearing comment, and
``quaid-sim-cpp/src/mqtt_controller.cpp:235`` for the publication side.
"""

import pytest

from quaid_env.config import Settings
from quaid_env.mqtt_controller import MqttController
from quaid_env.packets import MocapPacket, ObservationPacket, encode_mocap, encode_observation


def _make_controller(*, sim: bool = True) -> MqttController:
    settings = Settings()
    settings.robot.sim = sim
    settings.ports.mqtt_queue_no = '0'
    # Build a controller without connecting — we only exercise the
    # _handle_* paths which don't require the network.
    return MqttController(settings)


# -------------------------------------------------------- mocap units ----
def test_mocap_sim_keeps_raw_decimetre_units():
    """Sim mode: the controller must NOT divide pkt.x — the unit is dm and
    that's what the reward / z-center configs expect."""
    ctrl = _make_controller(sim=True)
    pkt = MocapPacket(
        header=0x0E, degrees=False, rigid_body_no=0,
        x=6, y=-3, z=16,                    # 0.6 m, -0.3 m, 1.6 m in dm
        yaw=0.1, pitch=0.0, roll=0.0,
        qr=1.0, qi=0.0, qj=0.0, qk=0.0,
    )
    ctrl._handle_mocap(encode_mocap(pkt))

    snap = ctrl.data.snapshot()
    # Raw int16 values pass through as-is in sim mode.
    assert snap.position_x == pytest.approx(6.0)
    assert snap.position_y == pytest.approx(-3.0)
    assert snap.position_z == pytest.approx(16.0)
    # no_rotation_* mirror the same unit (theta defaults to 0).
    assert snap.no_rotation_x == pytest.approx(6.0)
    assert snap.no_rotation_y == pytest.approx(-3.0)
    # yaw is float and stored unchanged (rotation by theta=0 is a no-op).
    assert snap.yaw == pytest.approx(0.1)
    assert snap.no_rotation_yaw == pytest.approx(0.1)


def test_mocap_real_divides_by_2_5_to_match_sim_units():
    """Real-mode mocap publishes 2.5x larger int16s than the simulator. The
    controller must /2.5 so the same reward config works in both modes —
    matches QuaidControllerMqtt.cpp:217-222."""
    ctrl = _make_controller(sim=False)
    pkt = MocapPacket(
        header=0x0E, degrees=False, rigid_body_no=0,
        x=15, y=-25, z=40,                  # 6 dm, -10 dm, 16 dm after /2.5
        yaw=0.0, pitch=0.0, roll=0.0,
        qr=1.0, qi=0.0, qj=0.0, qk=0.0,
    )
    ctrl._handle_mocap(encode_mocap(pkt))

    snap = ctrl.data.snapshot()
    assert snap.position_x == pytest.approx(15 / 2.5)   # 6.0 dm
    assert snap.position_y == pytest.approx(-25 / 2.5)  # -10.0 dm
    assert snap.position_z == pytest.approx(40 / 2.5)   # 16.0 dm


# -------------------------------------------------- observation packet ---
def test_observation_packet_populates_quaid_data():
    """Sanity check that obs handler maps every packet field into QuaidData
    without unit conversion (acc / current units stay as published)."""
    ctrl = _make_controller(sim=True)
    pkt = ObservationPacket(
        header=0x0A, time_delta=5,
        distance=0.0, yaw=0.1, pitch=0.05, roll=-0.05,
        voltage=12.3, current=2.0,
        position_knee_front_left=10, position_thigh_front_left=20,
        position_knee_front_right=30, position_thigh_front_right=40,
        position_knee_back_left=50, position_thigh_back_left=60,
        position_knee_back_right=70, position_thigh_back_right=80,
        current_front_left=0.5, current_front_right=0.6,
        current_back_left=0.7, current_back_right=0.8,
        acc_x=0.1, acc_y=0.2, acc_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
    )
    ctrl._handle_observation(encode_observation(pkt))

    snap = ctrl.data.snapshot()
    assert snap.yaw == pytest.approx(0.1)
    assert snap.pitch == pytest.approx(0.05)
    assert snap.current == pytest.approx(2.0)
    assert snap.acc_z == pytest.approx(9.81)
    assert snap.position_knee_back_left == 50
    assert snap.current_back_right == pytest.approx(0.8)
