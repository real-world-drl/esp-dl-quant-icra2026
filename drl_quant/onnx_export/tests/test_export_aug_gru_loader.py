"""Tests for the auto-detect / prefix-strip / bias-detect logic in
``drl_quant.onnx_export.export_aug_gru``.

These exist because the entry point is the one most likely to be hit by
external users (language-model researchers etc.) who didn't read any of
the rest of this repo's docs. Failure modes need to be obvious — a wrong
prefix would silently produce a quantized GRU with random weights, which
is much worse than a clean error.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from drl_quant.onnx_export.export_aug_gru import (
    _detect_dims,
    _find_gru,
    _load_state_dict,
    _remap_to_grunet,
    export,
    get_args,
)


# -------------------------------------------------- _find_gru -----------
def test_find_gru_bare_module():
    """A directly-saved nn.GRU has no prefix."""
    gru = nn.GRU(input_size=10, hidden_size=8, num_layers=2, bias=True)
    sd = gru.state_dict()
    prefix, n, has_bias = _find_gru(sd)
    assert prefix == ''
    assert n == 2
    assert has_bias is True


def test_find_gru_inside_wrapper():
    """A GRU buried inside an encoder gets a prefix like 'encoder.gru.'."""
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=4, hidden_size=8, num_layers=1, bias=False)

    sd = Encoder().state_dict()
    prefix, n, has_bias = _find_gru(sd)
    assert prefix == 'gru.'
    assert n == 1
    assert has_bias is False


def test_find_gru_nested_wrapper():
    """Two levels deep: 'encoder.rnn.weight_ih_l0' -> prefix='encoder.rnn.'."""
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(input_size=3, hidden_size=5, num_layers=3, bias=True)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Inner()

    prefix, n, has_bias = _find_gru(Outer().state_dict())
    assert prefix == 'encoder.rnn.'
    assert n == 3
    assert has_bias is True


def test_find_gru_no_gru_raises():
    sd = nn.Linear(4, 8).state_dict()
    with pytest.raises(ValueError, match='No GRU weights'):
        _find_gru(sd)


def test_find_gru_multiple_grus_raises():
    """Two GRUs in the same module is ambiguous — bail with a clear msg."""
    class TwoHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.GRU(4, 8, num_layers=1)
            self.right = nn.GRU(4, 8, num_layers=1)

    with pytest.raises(ValueError, match='Multiple GRUs'):
        _find_gru(TwoHeads().state_dict())


# ----------------------------------------------- _detect_dims -----------
def test_detect_dims():
    gru = nn.GRU(input_size=42, hidden_size=64, num_layers=1)
    inp, hid = _detect_dims(gru.state_dict(), prefix='')
    assert inp == 42
    assert hid == 64


# -------------------------------------------- _remap_to_grunet ----------
def test_remap_renames_to_rl_gru_prefix():
    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo = nn.GRU(input_size=4, hidden_size=8, num_layers=2, bias=True)

    sd = Wrapped().state_dict()
    out = _remap_to_grunet(sd, prefix='foo.', num_layers=2, has_bias=True)
    expected = {
        'rl_gru.weight_ih_l0', 'rl_gru.weight_hh_l0',
        'rl_gru.weight_ih_l1', 'rl_gru.weight_hh_l1',
        'rl_gru.bias_ih_l0', 'rl_gru.bias_hh_l0',
        'rl_gru.bias_ih_l1', 'rl_gru.bias_hh_l1',
    }
    assert set(out.keys()) == expected
    # Values were copied through, not deep-copied — same tensor objects.
    torch.testing.assert_close(out['rl_gru.weight_ih_l0'], sd['foo.weight_ih_l0'])


# ------------------------------------------- _load_state_dict -----------
def test_load_state_dict_pickle(tmp_path):
    """torch.save(state_dict) — the typical Python ML save format."""
    gru = nn.GRU(input_size=4, hidden_size=8, num_layers=1)
    p = tmp_path / 'gru.pt'
    torch.save(gru.state_dict(), p)
    sd = _load_state_dict(str(p))
    assert 'weight_ih_l0' in sd


def test_load_state_dict_full_module(tmp_path):
    """torch.save(model) — less common but still valid."""
    gru = nn.GRU(input_size=4, hidden_size=8, num_layers=1)
    p = tmp_path / 'gru.pt'
    torch.save(gru, p)
    sd = _load_state_dict(str(p))
    assert 'weight_ih_l0' in sd


def test_load_state_dict_torchscript(tmp_path):
    """torch.jit.save — what the C++ training repo's .dat files are."""
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=4, hidden_size=8, num_layers=1)

        def forward(self, x, h):
            return self.gru(x, h)

    scripted = torch.jit.script(Wrap())
    p = tmp_path / 'gru.pt'
    torch.jit.save(scripted, p)
    sd = _load_state_dict(str(p))
    assert any(k.endswith('weight_ih_l0') for k in sd.keys())


# --------------------------------------------- end-to-end export --------
@pytest.mark.parametrize('layers, bias, input_size, hidden_size', [
    (1, True,  10, 16),
    (2, True,  20, 32),
    (3, False, 25, 64),    # matches the bundled Quaid GRU shape
    (1, False, 100, 128),  # typical language-model size
])
def test_end_to_end_python_trained_gru(tmp_path, layers, bias, input_size, hidden_size):
    """Save a Python-trained nn.GRU as a regular .pt, run the exporter
    against it, load the resulting ONNX, and confirm the output matches
    the original PyTorch GRU within float32 noise. This is the load-bearing
    test for the whole quickstart workflow."""
    torch.manual_seed(0)
    gru = nn.GRU(
        input_size=input_size, hidden_size=hidden_size,
        num_layers=layers, bias=bias, batch_first=True,
    ).eval()

    src_path = tmp_path / 'my_gru.pt'
    torch.save(gru.state_dict(), src_path)

    onnx_path = tmp_path / 'aug_my_gru.onnx'
    args = get_args(['-i', str(src_path), '-o', str(onnx_path)])
    export(args)

    # Run the exported ONNX against a fixed input and compare to nn.GRU.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))

    x = torch.randn(1, 1, input_size)
    h = torch.zeros(layers, 1, hidden_size)
    with torch.no_grad():
        ref_out, ref_h = gru(x, h)

    onnx_out, onnx_h = sess.run(
        ['features', 'h_t'],
        {
            'observations': x.squeeze(1).numpy().astype(np.float32),
            'h_t_in': h.numpy(),
        },
    )

    # ref_out is the GRU's last-layer output sequence: shape (1, 1, hidden).
    # The Aug-GRU exporter returns (1, hidden) for the last layer's hidden.
    # They should match the GRU's final hidden state.
    np.testing.assert_allclose(
        ref_h.numpy(), onnx_h, atol=1e-5,
        err_msg='hidden state mismatch — bias / cascade ordering must agree with nn.GRU',
    )
    np.testing.assert_allclose(
        ref_out.squeeze(1).numpy(), np.asarray(onnx_out).reshape(1, -1),
        atol=1e-5,
        err_msg='last-layer features mismatch',
    )


def test_unsupported_layer_count_raises(tmp_path):
    """A 4-layer GRU must fail clearly at GRUAug construction, not later."""
    gru = nn.GRU(input_size=10, hidden_size=8, num_layers=4, bias=False)
    src = tmp_path / 'gru.pt'
    torch.save(gru.state_dict(), src)
    args = get_args(['-i', str(src), '-o', str(tmp_path / 'out.onnx')])
    with pytest.raises(ValueError, match='1, 2, 3'):
        export(args)
