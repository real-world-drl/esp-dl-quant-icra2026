"""Equivalence tests: GRUAug vs torch.nn.GRU.

The whole point of `GRUAug` is to be **numerically identical** to
`torch.nn.GRU(num_layers=N, bias=False)` after layer-by-layer weight
transplant — so trained `nn.GRU` weights can be deployed via the
ESP-DL-friendly graph without any behaviour change. These tests lock that
equivalence in for layer counts 1, 2 and 3, both at a single timestep
(checks layer 0 separately from the inter-layer cascade) and over a
multi-step rollout (catches accumulating divergence).
"""

import pytest
import torch
import torch.nn as nn

from drl_quant.networks.augmented_gru import GRUAug, GRUCellAug


# --------------------------------------------------------------- helpers ---
def _transplant(gru_aug: GRUAug, gru: nn.GRU) -> None:
    """Copy weights from a trained nn.GRU into the matching GRUAug layers.

    Mirrors the loop the exporters use; centralised here so the test fails
    if the exporter weight-transplant convention drifts."""
    for i in range(gru_aug.layers):
        layer = getattr(gru_aug, f'l{i}')
        layer.x2h.weight = getattr(gru, f'weight_ih_l{i}')
        layer.h2h.weight = getattr(gru, f'weight_hh_l{i}')


def _build_pair(layers: int, input_size: int = 5, hidden_size: int = 7, *,
                seed: int = 0) -> tuple[nn.GRU, GRUAug]:
    torch.manual_seed(seed)
    gru = nn.GRU(
        input_size=input_size, hidden_size=hidden_size,
        num_layers=layers, batch_first=True, bias=False,
    )
    gru.eval()
    aug = GRUAug(input_size, hidden_size, bias=False, layers=layers)
    _transplant(aug, gru)
    aug.eval()
    return gru, aug


# --------------------------------------------------- single-cell sanity ---
def test_grucellaug_matches_native_grucell():
    """GRUCellAug must be bit-for-bit equivalent to nn.GRUCell with bias=False —
    this is the foundation that the multi-layer wrapper builds on."""
    torch.manual_seed(0)
    cell_native = nn.GRUCell(5, 7, bias=False)
    cell_aug = GRUCellAug(5, 7, bias=False)
    cell_aug.x2h.weight = cell_native.weight_ih
    cell_aug.h2h.weight = cell_native.weight_hh

    x = torch.randn(1, 5)
    h = torch.randn(1, 7)
    with torch.no_grad():
        out_native = cell_native(x, h)
        out_aug = cell_aug(x, h)
    assert torch.allclose(out_native, out_aug, atol=1e-6)


# ---------------------------------------------------- equivalence tests ---
@pytest.mark.parametrize('layers', [1, 2, 3])
def test_gruaug_matches_native_gru_single_step(layers):
    """Single forward pass at h=0. Catches cascade bugs immediately because
    the buggy delayed-cascade variant produces all-zero outputs at h=0 for
    layers >= 1, while the proper cascade does not."""
    gru, aug = _build_pair(layers)
    x = torch.randn(1, 1, 5)         # (batch, seq=1, input)
    h = torch.zeros(layers, 1, 7)

    with torch.no_grad():
        out_native, h_native = gru(x, h)
        out_aug, h_aug = aug(x.squeeze(1), h)

    # Output of nn.GRU is the last layer's hidden state (with the seq dim).
    assert torch.allclose(out_native.squeeze(1), out_aug, atol=1e-5), \
        f'output mismatch for layers={layers}'
    assert torch.allclose(h_native, h_aug, atol=1e-5), \
        f'final hidden state mismatch for layers={layers}'


@pytest.mark.parametrize('layers', [1, 2, 3])
def test_gruaug_matches_native_gru_rollout(layers):
    """50-step rollout with random inputs and non-zero h. Errors that look
    small at t=0 (e.g. tail layers using h=0 instead of the cascade) blow
    up over time, so this is the strongest check."""
    gru, aug = _build_pair(layers, seed=layers)

    h_native = torch.randn(layers, 1, 7)
    h_aug = h_native.clone()
    max_diff = 0.0
    for t in range(50):
        x = torch.randn(1, 1, 5)
        with torch.no_grad():
            out_native, h_native = gru(x, h_native)
            out_aug, h_aug = aug(x.squeeze(1), h_aug)
        max_diff = max(max_diff, (out_native.squeeze(1) - out_aug).abs().max().item())

    assert max_diff < 1e-5, f'max divergence over 50 steps for layers={layers}: {max_diff}'


# ------------------------------------------------------- shape / errors ---
@pytest.mark.parametrize('layers', [1, 2, 3])
def test_hidden_state_shape_round_trip(layers):
    """Output hidden state shape should match the input hidden state shape
    for every supported layer count. Important because the runner threads
    h_t back into the next call without reshaping."""
    gru, aug = _build_pair(layers)
    x = torch.randn(1, 5)
    h = torch.zeros(layers, 1, 7)
    with torch.no_grad():
        _, h_out = aug(x, h)
    assert h_out.shape == h.shape


@pytest.mark.parametrize('bad_layers', [0, 4, 5, -1])
def test_unsupported_layer_count_raises(bad_layers):
    with pytest.raises(ValueError, match='supports'):
        GRUAug(5, 7, layers=bad_layers)


# ----------------------------------------------- transplant via exporter ---
def test_exporter_loop_matches_explicit_assignment():
    """The exporters use a `for i in range(num_layers)` loop with getattr
    instead of explicit ``aug.l0.x2h.weight = gru.weight_ih_l0`` etc.
    Make sure both produce the same end state.
    """
    torch.manual_seed(0)
    gru = nn.GRU(input_size=5, hidden_size=7, num_layers=3, batch_first=True, bias=False)

    aug_loop = GRUAug(5, 7, layers=3)
    for i in range(3):
        layer = getattr(aug_loop, f'l{i}')
        layer.x2h.weight = getattr(gru, f'weight_ih_l{i}')
        layer.h2h.weight = getattr(gru, f'weight_hh_l{i}')

    aug_explicit = GRUAug(5, 7, layers=3)
    aug_explicit.l0.x2h.weight = gru.weight_ih_l0
    aug_explicit.l0.h2h.weight = gru.weight_hh_l0
    aug_explicit.l1.x2h.weight = gru.weight_ih_l1
    aug_explicit.l1.h2h.weight = gru.weight_hh_l1
    aug_explicit.l2.x2h.weight = gru.weight_ih_l2
    aug_explicit.l2.h2h.weight = gru.weight_hh_l2

    x = torch.randn(1, 5)
    h = torch.randn(3, 1, 7)
    with torch.no_grad():
        out_loop, h_loop = aug_loop(x, h)
        out_explicit, h_explicit = aug_explicit(x, h)
    assert torch.allclose(out_loop, out_explicit, atol=1e-7)
    assert torch.allclose(h_loop, h_explicit, atol=1e-7)
