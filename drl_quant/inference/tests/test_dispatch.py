"""Filename-based auto-dispatch tests.

These do not load any actual models — they just exercise the rules from
``Player.cpp:27-95``. If a future training run uses a new naming convention,
add a case here so the dispatch stays predictable.
"""

import pytest

from drl_quant.inference.player import detect_policy_variant


@pytest.mark.parametrize('name, expected', [
    # Non-recurrent TD3 / SAC ONNX -- single input, no preprocessor needed.
    ('act_net_QuaidSIM-v4_TD3_+225.827_750000.onnx', {
        'format': 'onnx', 'has_gru_inside': False,
        'is_recurrent': False, 'actions_to_rnn': False, 'algorithm': 'TD3',
    }),
    ('act_net_QuaidSIM-v4_SAC_+238.823_775000.onnx', {
        'format': 'onnx', 'has_gru_inside': False,
        'is_recurrent': False, 'actions_to_rnn': False, 'algorithm': 'SAC',
    }),

    # Aug-GRU ONNX with R- (no prev action) -- runner manages h_t internally,
    # observation flows in unchanged.
    ('aug_act_net_QuaidSIM-v4_R-TD3_+0.000_0.onnx', {
        'format': 'onnx', 'has_gru_inside': True,
        'is_recurrent': True, 'actions_to_rnn': False, 'algorithm': 'TD3',
    }),

    # Aug-GRU ONNX with RA- -- runner manages h_t, AddActionsPreprocessor
    # prepends prev_action so the input is 25-dim.
    ('aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx', {
        'format': 'onnx', 'has_gru_inside': True,
        'is_recurrent': True, 'actions_to_rnn': True, 'algorithm': 'TD3',
    }),
    ('aug_act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.onnx', {
        'format': 'onnx', 'has_gru_inside': True,
        'is_recurrent': True, 'actions_to_rnn': True, 'algorithm': 'SAC',
    }),

    # Native-GRU baked in -- same dispatch as Aug.
    ('with_gru_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx', {
        'format': 'onnx', 'has_gru_inside': True,
        'is_recurrent': True, 'actions_to_rnn': True, 'algorithm': 'TD3',
    }),

    # Dynamic-quantised ONNX -- the _qd suffix doesn't change anything.
    ('aug_act_net_QuaidSIM-v4_RA-SAC_+299.788_850000_qd.onnx', {
        'format': 'onnx', 'has_gru_inside': True,
        'is_recurrent': True, 'actions_to_rnn': True, 'algorithm': 'SAC',
    }),

    # TorchScript actors -- format flips to 'traced'.
    ('act_net_QuaidSIM-v4_TD3_+225.827_750000.dat', {
        'format': 'traced', 'has_gru_inside': False,
        'is_recurrent': False, 'actions_to_rnn': False, 'algorithm': 'TD3',
    }),
    ('act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.dat', {
        'format': 'traced', 'has_gru_inside': False,
        'is_recurrent': True, 'actions_to_rnn': True, 'algorithm': 'SAC',
    }),
    ('actor.pt', {
        'format': 'traced', 'has_gru_inside': False,
        'is_recurrent': False, 'actions_to_rnn': False, 'algorithm': None,
    }),
])
def test_detect_policy_variant(name, expected):
    assert detect_policy_variant(name) == expected


def test_detect_policy_variant_rejects_unknown_extension():
    with pytest.raises(ValueError):
        detect_policy_variant('model.bin')


def test_R_and_RA_dont_collide():
    """The 'R-' substring lives inside 'RA-'; make sure RA- doesn't get
    misclassified as R-only."""
    info = detect_policy_variant('aug_act_net_QuaidSIM-v4_RA-TD3_+0_0.onnx')
    assert info['is_recurrent'] is True
    assert info['actions_to_rnn'] is True  # the bit that distinguishes RA from R
