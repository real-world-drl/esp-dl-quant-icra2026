"""Tests for the filename-based algorithm detection used by the exporters."""

import pytest

from drl_quant.onnx_export._naming import detect_algorithm


@pytest.mark.parametrize('name, expected', [
    ('act_net_QuaidSIM-v4_TD3_+225.827_750000.dat', 'TD3'),
    ('act_net_QuaidSIM-v4_SAC_+238.823_775000.dat', 'SAC'),
    ('act_net_QuaidSIM-v4_R-TD3_+364.117_475000.dat', 'TD3'),
    ('act_net_QuaidSIM-v4_R-SAC_+0.000_0.dat', 'SAC'),
    ('act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat', 'TD3'),
    ('act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.dat', 'SAC'),
    ('/full/path/to/act_net_TD3.dat', 'TD3'),
])
def test_detect_returns_algorithm(name, expected):
    assert detect_algorithm(name) == expected


def test_unknown_filename_raises_with_examples():
    """The error message should be actionable: it has to name the offending
    file, list valid examples, and suggest a fix. We assert each of those
    pieces is present so future edits keep the message helpful."""
    bad = 'act_net_Ramp+301.146.dat'
    with pytest.raises(ValueError) as exc:
        detect_algorithm(bad)
    msg = str(exc.value)

    assert bad in msg                      # names the offending file
    assert 'TD3' in msg and 'SAC' in msg   # mentions both algorithms
    assert 'rename' in msg.lower()         # suggests an action
    assert 'act_net_QuaidSIM-v4_TD3' in msg  # at least one valid example
    assert 'mv act_net_Ramp' in msg        # concrete fix command
