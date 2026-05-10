"""Tests for the run-output folder convention + filename autodetection."""

import pytest

from drl_quant.inference.__main__ import (
    build_run_dir,
    detect_env_name,
    detect_policy_name,
)


# ---------------------------------------------------- env name parsing ---
@pytest.mark.parametrize('path, expected', [
    ('act_net_QuaidSIM-v4_TD3_+225.827_750000.dat', 'QuaidSIM-v4'),
    ('aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx', 'QuaidSIM-v4'),
    ('with_gru_act_net_QuaidSIM-v4_RA-SAC_+0.000_0.onnx', 'QuaidSIM-v4'),
    ('models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_SAC_+1.dat', 'QuaidSIM-v4'),
    # Different env name still works.
    ('aug_act_net_PendulumSim_TD3_+1.onnx', 'PendulumSim'),
    # Filename without the convention falls back gracefully.
    ('totally_unrelated_file.onnx', 'unknown'),
])
def test_detect_env_name(path, expected):
    assert detect_env_name(path) == expected


# ------------------------------------------------- policy name parsing ---
@pytest.mark.parametrize('path, expected', [
    # Non-recurrent — bare algorithm name.
    ('act_net_QuaidSIM-v4_TD3_+1.dat', 'TD3'),
    ('act_net_QuaidSIM-v4_SAC_+1.dat', 'SAC'),
    ('act_net_QuaidSIM-v4_TD3_+1.onnx', 'TD3'),

    # External-GRU TorchScript: R-/RA- prefix preserved.
    ('act_net_QuaidSIM-v4_R-TD3_+1.dat', 'R-TD3'),
    ('act_net_QuaidSIM-v4_RA-SAC_+1.dat', 'RA-SAC'),

    # Aug-GRU baked in (aug_*, with_gru_*, with_rnn_*) — A- prefix
    # regardless of whether the source name was R- or RA-.
    ('aug_act_net_QuaidSIM-v4_R-TD3_+1.onnx', 'A-TD3'),
    ('aug_act_net_QuaidSIM-v4_RA-TD3_+1.onnx', 'A-TD3'),
    ('aug_act_net_QuaidSIM-v4_RA-SAC_+1.onnx', 'A-SAC'),
    ('with_gru_act_net_QuaidSIM-v4_RA-TD3_+1.onnx', 'A-TD3'),
    ('with_gru_act_net_QuaidSIM-v4_RA-SAC_+1.onnx', 'A-SAC'),

    # Dynamic-quant suffix doesn't alter the policy classification.
    ('aug_act_net_QuaidSIM-v4_RA-TD3_+1_qd.onnx', 'A-TD3'),
])
def test_detect_policy_name(path, expected):
    assert detect_policy_name(path) == expected


# ----------------------------------------------------- run dir layout ---
def test_build_run_dir_creates_three_levels(tmp_path):
    """Mirrors HyperParams::init_snapshot_dir from the C++ project:
    <root>/<env>/<policy>/<timestamp>/. The dir must exist after the call
    so the env / stats logger can write into it without further mkdir."""
    out = build_run_dir(
        output_root=str(tmp_path),
        env_name='QuaidSIM-v4',
        policy_name='A-TD3',
        timestamp='2026-05-10T15-32-28',
    )
    assert out.exists()
    assert out.is_dir()
    # Path layout exactly matches the C++ convention.
    assert out == tmp_path / 'QuaidSIM-v4' / 'A-TD3' / '2026-05-10T15-32-28'


def test_build_run_dir_idempotent(tmp_path):
    """A second call with the same args must succeed (mkdir parents=True,
    exist_ok=True). C++ has the same "reuse existing" semantics."""
    args = dict(output_root=str(tmp_path), env_name='E', policy_name='P',
                timestamp='T')
    out1 = build_run_dir(**args)
    out2 = build_run_dir(**args)
    assert out1 == out2
