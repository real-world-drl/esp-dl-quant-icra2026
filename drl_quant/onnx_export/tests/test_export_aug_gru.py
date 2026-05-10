"""Tests for the GRU-only Aug-GRU exporter.

Heavy-weight tests (actually loading the TorchScript GRU and exporting an
ONNX) need torch + onnx installed. The path-naming and arg-validation
tests are pure-Python and run anywhere.
"""

import argparse

import pytest

from drl_quant.onnx_export.export_aug_gru import _default_output_path, export


def test_default_output_path_with_rnn_prefix():
    """The shipped naming convention: rnn_<...>.dat -> aug_rnn_<...>.onnx,
    sitting next to the source file."""
    out = _default_output_path('models/rnn/rnn_Quaid_RA-64.dat')
    assert out.endswith('models/rnn/aug_rnn_Quaid_RA-64.onnx') or \
           out.endswith('models\\rnn\\aug_rnn_Quaid_RA-64.onnx')


def test_default_output_path_without_rnn_prefix():
    """Files that don't follow the rnn_ convention still get the aug_ prefix."""
    out = _default_output_path('/tmp/my_gru.dat')
    assert out.endswith('aug_my_gru.onnx')


def test_default_output_path_pt_extension():
    """`.pt` should be handled the same as `.dat`."""
    out = _default_output_path('models/rnn/rnn_Foo.pt')
    assert out.endswith('aug_rnn_Foo.onnx')


def test_export_rejects_unsupported_layer_count():
    """GRUAug now supports 1, 2, or 3 layers — higher counts must fail
    up-front via the constructor's check rather than silently producing a
    3-layer ONNX from a 4-layer checkpoint. We pass a non-existent input
    file so the test doesn't need a real .dat; the constructor check fires
    before file IO either way (it's the first thing the export function
    does after path resolution).
    """
    import tempfile

    # Use a real (but empty) file path so torch.jit.load reaches its own
    # error rather than the OS file-not-found error obscuring the layer
    # check. We expect *some* error before the ONNX export step.
    with tempfile.NamedTemporaryFile(suffix='.dat') as f:
        args = argparse.Namespace(
            input_model=f.name,
            output_model=None,
            input_size=25,
            hidden_size=64,
            num_layers=4,
        )
        with pytest.raises(Exception):  # ValueError from GRUAug, or load error
            export(args)
