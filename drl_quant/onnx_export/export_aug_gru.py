"""Step 2d: export a **GRU-only** Aug-GRU ONNX (no actor head).

Accepts any of the following as input:

* TorchScript ``.dat`` from the upstream C++ training repo (e.g.
  ``models/rnn/rnn_Quaid_RA-64.dat``).
* Pure-Python ``torch.save(model.state_dict(), 'gru.pt')`` — what most
  language-model / sequence-encoder workflows produce.
* Full ``nn.Module`` pickles (``torch.save(model, 'gru.pt')``).

Input dim, hidden size, layer count and bias presence are auto-detected
from the state_dict — pass them on the CLI only when you want to override
the autodetection. The wrapper prefix (``rnn.``, ``gru.``, ``rl_gru.``,
or none) is stripped automatically too, so a GRU buried inside a larger
encoder module loads cleanly.

ESP-DL cannot quantize the standard ONNX GRU op, so we go through Aug-GRU
just like the actor exporter does. The resulting ONNX has two inputs
(``observations``, ``h_t_in``) and two outputs (``features``, ``h_t``)
and feeds straight into:

* ``drl_quant.onnx_dynamic_quantize.quantize`` for an int8 host-side benchmark;
* ``drl_quant.espdl_quantize.quantize_recurrent`` for the ESP-DL deployment
  bundle;
* ``drl_quant.inference.preprocessors.OnnxGruPreprocessor`` if you want to
  use the quantized GRU as a feature extractor in front of a separate actor.

Output filename follows the ``aug_`` prefix convention used by the actor
exporters: ``rnn_<...>.dat`` -> ``aug_rnn_<...>.onnx``.

For language-model and sequence-encoder users, see ``GRU_QUICKSTART.md``
at the project root for the full end-to-end walkthrough.

Run as::

    # Auto-detect everything from the checkpoint:
    python -m drl_quant.onnx_export.export_aug_gru -i models/rnn/rnn_Quaid_RA-64.dat

    # Or override specific dims if you don't trust autodetection:
    python -m drl_quant.onnx_export.export_aug_gru -i my_gru.pt -hs 128 --no-bias
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch

from drl_quant.networks.augmented_gru import GRUAug
from drl_quant.networks.rnn import GruNet
from drl_quant.onnx_export._export_compat import LEGACY_EXPORT_KWARGS


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def _load_state_dict(path: str) -> dict:
    """Get a state_dict out of any reasonable PyTorch serialisation.

    Tries TorchScript first (works for the C++ training repo's ``.dat``
    output archives that have weights but no forward graph), then falls
    back to ``torch.load`` for regular pickles. Bare state_dict pickles,
    full-module pickles, and nested module pickles all work.
    """
    try:
        scripted = torch.jit.load(path, map_location='cpu')
        return scripted.state_dict()
    except RuntimeError:
        pass  # not TorchScript, try regular pickle

    loaded = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(loaded, dict):
        return loaded
    if hasattr(loaded, 'state_dict'):
        return loaded.state_dict()
    raise ValueError(
        f'Could not extract a state_dict from {path}: got {type(loaded).__name__}'
    )


def _find_gru(sd: dict) -> tuple[str, int, bool]:
    """Locate the GRU submodule inside ``sd`` and report its layout.

    Returns ``(prefix, num_layers, has_bias)``. ``prefix`` is everything
    before ``weight_ih_l0`` in the source key — e.g. ``'rl_gru.'``,
    ``'rnn.'``, ``'encoder.gru.'``, or ``''`` for a bare ``nn.GRU``.
    """
    prefixes = sorted({
        k[: -len('weight_ih_l0')]
        for k in sd.keys() if k.endswith('weight_ih_l0')
    })
    if not prefixes:
        raise ValueError(
            'No GRU weights found in checkpoint — expected at least one key '
            "ending in 'weight_ih_l0'. Got top-level keys: "
            f'{sorted({k.split(".")[0] for k in sd.keys()})}'
        )
    if len(prefixes) > 1:
        raise ValueError(
            'Multiple GRUs found in the checkpoint, cannot disambiguate. '
            f'Prefixes: {prefixes}. Save the specific GRU submodule on its '
            'own (`torch.save(model.gru.state_dict(), ...)`) and re-run.'
        )
    prefix = prefixes[0]

    # Count layers by walking weight_ih_l<N> until the chain breaks.
    n = 0
    while f'{prefix}weight_ih_l{n}' in sd:
        n += 1

    has_bias = f'{prefix}bias_ih_l0' in sd
    return prefix, n, has_bias


def _detect_dims(sd: dict, prefix: str) -> tuple[int, int]:
    """Read ``input_dim`` and ``hidden_size`` off the first layer's weight.

    For ``nn.GRU``, ``weight_ih_l0`` has shape ``(3 * hidden_size, input_dim)``.
    """
    w = sd[f'{prefix}weight_ih_l0']
    out_dim, input_dim = w.shape
    if out_dim % 3 != 0:
        raise ValueError(
            f'weight_ih_l0 has shape {tuple(w.shape)}; first dim must be '
            'divisible by 3 (= 3 * hidden_size).'
        )
    return int(input_dim), int(out_dim // 3)


def _remap_to_grunet(sd: dict, prefix: str, num_layers: int, has_bias: bool) -> dict:
    """Rename source keys to match ``GruNet``'s ``rl_gru.*`` prefix."""
    out = {}
    weight_keys = ('weight_ih', 'weight_hh')
    bias_keys = ('bias_ih', 'bias_hh') if has_bias else ()
    for layer in range(num_layers):
        for key in weight_keys + bias_keys:
            src = f'{prefix}{key}_l{layer}'
            dst = f'rl_gru.{key}_l{layer}'
            out[dst] = sd[src]
    return out


# ---------------------------------------------------------------------------
def get_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '-i', '--input_model', required=True,
        help='Path to the GRU checkpoint. Accepts TorchScript .dat / .pt or '
             'a regular torch.save() pickle of an nn.Module / state_dict.',
    )
    parser.add_argument('-o', '--output_model', help='Override the output ONNX path.')
    parser.add_argument(
        '-n', '--input_size', type=int, default=None,
        help='Override GRU input dimension. Auto-detected from the checkpoint when omitted.',
    )
    parser.add_argument(
        '-hs', '--hidden_size', type=int, default=None,
        help='Override GRU hidden size. Auto-detected when omitted.',
    )
    parser.add_argument(
        '-l', '--num_layers', type=int, default=None,
        help='Override GRU layer count (1-3). Auto-detected when omitted.',
    )
    parser.add_argument(
        '--bias', action=argparse.BooleanOptionalAction, default=None,
        help='Whether the GRU has bias terms. Auto-detected when omitted. '
             'Use --no-bias to force a bias-free model (the C++ training repo '
             'default); --bias for the nn.GRU default of bias=True.',
    )
    return parser.parse_args(argv)


def _default_output_path(input_path: str) -> str:
    """``models/rnn/rnn_Quaid_RA-64.dat`` -> ``models/rnn/aug_rnn_Quaid_RA-64.onnx``.

    Matches the convention the C++ training repo's ``aug_rnn_*`` artefacts
    use, and avoids clobbering any existing native-GRU ONNX export sitting
    next to the source ``.dat``.
    """
    p = Path(input_path)
    name = p.name
    if name.endswith('.dat'):
        stem = name[: -len('.dat')]
    elif name.endswith('.pt'):
        stem = name[: -len('.pt')]
    else:
        stem = p.stem
    return str(p.with_name('aug_' + stem + '.onnx'))


def export(args: argparse.Namespace) -> str:
    if not args.output_model:
        args.output_model = _default_output_path(args.input_model)

    log.info('loading GRU from %s', args.input_model)
    sd = _load_state_dict(args.input_model)
    prefix, det_layers, det_bias = _find_gru(sd)
    det_input, det_hidden = _detect_dims(sd, prefix)

    # CLI args override autodetection. Warn loudly if they disagree so
    # a misclick on `--hidden_size 128` against a 64-hidden checkpoint
    # doesn't silently produce a model that load_state_dict will reject.
    input_size = args.input_size or det_input
    hidden_size = args.hidden_size or det_hidden
    num_layers = args.num_layers or det_layers
    has_bias = det_bias if args.bias is None else args.bias

    for name, override, detected in (
        ('input_size', input_size, det_input),
        ('hidden_size', hidden_size, det_hidden),
        ('num_layers', num_layers, det_layers),
        ('bias', has_bias, det_bias),
    ):
        if override != detected:
            log.warning('--%s=%s overrides detected value %s', name, override, detected)

    log.info(
        'GRU layout: prefix=%r, input_size=%d, hidden_size=%d, layers=%d, bias=%s',
        prefix, input_size, hidden_size, num_layers, has_bias,
    )

    # Build a Python GruNet with matching shape and load the renamed weights.
    params = {'rnn_hidden_size': hidden_size, 'rnn_layers': num_layers, 'bias': has_bias}
    native_gru = GruNet(input_size, params)
    native_gru.load_state_dict(_remap_to_grunet(sd, prefix, num_layers, has_bias))
    native_gru.eval()

    # Transplant into Aug-GRU. The Aug-GRU constructor rejects layer counts
    # outside 1-3 (its forward branches per layer count) so users with
    # >3-layer GRUs get a clear error here rather than later.
    gru_aug = GRUAug(input_size, hidden_size, bias=has_bias, layers=num_layers)
    for i in range(num_layers):
        layer = getattr(gru_aug, f'l{i}')
        layer.x2h.weight = getattr(native_gru.rl_gru, f'weight_ih_l{i}')
        layer.h2h.weight = getattr(native_gru.rl_gru, f'weight_hh_l{i}')
        if has_bias:
            layer.x2h.bias = getattr(native_gru.rl_gru, f'bias_ih_l{i}')
            layer.h2h.bias = getattr(native_gru.rl_gru, f'bias_hh_l{i}')
    gru_aug.eval()

    dummy_obs = torch.zeros(1, input_size, dtype=torch.float32)
    dummy_h = torch.zeros(num_layers, 1, hidden_size)

    out_dir = os.path.dirname(args.output_model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            gru_aug,
            (dummy_obs, dummy_h),
            args.output_model,
            verbose=False,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['observations', 'h_t_in'],
            output_names=['features', 'h_t'],
            **LEGACY_EXPORT_KWARGS,
        )
    log.info('exported ONNX to %s', args.output_model)
    return args.output_model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    export(get_args())


if __name__ == '__main__':
    main()
