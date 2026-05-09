"""Round-trip tests for the binary packet codecs."""

import pytest

from quaid_env.packets import (
    MOCAP_HEADER,
    MOCAP_STRUCT,
    OBS_STRUCT,
    OBSERVATION_HEADER,
    MocapPacket,
    ObservationPacket,
    decode_mocap,
    decode_observation,
    encode_mocap,
    encode_observation,
)


def test_observation_struct_size_matches_cpp():
    # The C++ struct is __attribute__((packed)); on x86/ARM that's a flat
    # 1 + 2 + 6*4 + 8*2 + 4*4 + 6*4 = 83 bytes. Lock that in.
    assert OBS_STRUCT.size == 83


def test_mocap_struct_size_matches_cpp():
    # 1 + 1 + 1 + 3*2 + 7*4 = 37
    assert MOCAP_STRUCT.size == 37


def _make_obs():
    return ObservationPacket(
        header=OBSERVATION_HEADER,
        time_delta=42,
        distance=1.5, yaw=-0.25, pitch=0.05, roll=-0.05,
        voltage=12.3, current=2.1,
        position_knee_front_left=100, position_thigh_front_left=110,
        position_knee_front_right=120, position_thigh_front_right=130,
        position_knee_back_left=140, position_thigh_back_left=150,
        position_knee_back_right=160, position_thigh_back_right=170,
        current_front_left=0.5, current_front_right=0.6,
        current_back_left=0.7, current_back_right=0.8,
        acc_x=0.1, acc_y=0.2, acc_z=9.8,
        gyro_x=0.01, gyro_y=0.02, gyro_z=0.03,
    )


def _make_mocap():
    return MocapPacket(
        header=MOCAP_HEADER,
        degrees=False,
        rigid_body_no=2,
        x=1234, y=-567, z=890,
        yaw=0.5, pitch=-0.1, roll=0.0,
        qr=1.0, qi=0.0, qj=0.0, qk=0.0,
    )


def _assert_packet_close(decoded, original):
    """Compare field-by-field — float fields lose precision via the 32-bit
    `f` format, so they need approx equality; ints are exact."""
    for f in original.__dataclass_fields__:
        a, b = getattr(decoded, f), getattr(original, f)
        if isinstance(b, float):
            assert a == pytest.approx(b, abs=1e-5, rel=1e-6), f'field {f!r} differs'
        else:
            assert a == b, f'field {f!r} differs'


def test_observation_round_trip():
    pkt = _make_obs()
    payload = encode_observation(pkt)
    assert len(payload) == 83
    decoded = decode_observation(payload)
    _assert_packet_close(decoded, pkt)


def test_mocap_round_trip():
    pkt = _make_mocap()
    payload = encode_mocap(pkt)
    assert len(payload) == 37
    decoded = decode_mocap(payload)
    _assert_packet_close(decoded, pkt)


def test_observation_rejects_wrong_header():
    pkt = _make_obs()
    payload = bytearray(encode_observation(pkt))
    payload[0] = 0xFF
    with pytest.raises(ValueError):
        decode_observation(bytes(payload))


def test_mocap_rejects_wrong_header():
    pkt = _make_mocap()
    payload = bytearray(encode_mocap(pkt))
    payload[0] = 0xFF
    with pytest.raises(ValueError):
        decode_mocap(bytes(payload))
