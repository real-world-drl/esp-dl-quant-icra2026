"""Step 2d: export a **GRU-only** Aug-GRU ONNX (no actor head).

The TorchScript ``.dat`` from the upstream training repo (e.g.
``models/rnn/rnn_Quaid_RA-64.dat``) holds an ``nn.GRU`` checkpoint. ESP-DL
cannot quantize the standard ONNX GRU op, so we go through Aug-GRU just like
the actor exporter does — the weight layout matches between the two so we
copy ``weight_ih_l<N>`` and ``weight_hh_l<N>`` straight across.

This entry point is here for **language-model and other GRU-only use cases**
where you don't have an actor head and just want a quantized GRU. The
resulting ONNX has two inputs (``observations``, ``h_t_in``) and two outputs
(``features``, ``h_t``) and feeds straight into:

* ``drl_quant.onnx_dynamic_quantize.quantize`` for an int8 host-side benchmark;
* ``drl_quant.espdl_quantize.quantize_recurrent`` for the ESP-DL deployment
  bundle (the same calibration loader works because the obs / h_t input shape
  is unchanged from the actor case);
* ``drl_quant.inference.preprocessors.OnnxGruPreprocessor`` if you want to
  use the quantized GRU as a feature extractor in front of a separate actor.

Output filename follows the ``aug_`` prefix convention used by the actor
exporters: ``rnn_<...>.dat`` -> ``aug_rnn_<...>.onnx``.

Run as::

    python -m drl_quant.onnx_export.export_aug_gru \\
        -i models/rnn/rnn_Quaid_RA-64.dat \\
        -n 25                                 # 17 obs + 8 prev_action for RA-
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from drl_quant.networks.augmented_gru import GRUAug
from drl_quant.networks.rnn import GruNet
from drl_quant.onnx_export._export_compat import LEGACY_EXPORT_KWARGS


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input_model', required=True, help='TorchScript GRU .dat.')
    parser.add_argument('-o', '--output_model', help='Override the output ONNX path.')
    parser.add_argument(
        '-n', '--input_size', type=int, required=True,
        help='GRU input dimension. For Quaid: 17 (R- variants), 25 (RA- variants).',
    )
    parser.add_argument('-hs', '--hidden_size', default=64, type=int, help='GRU hidden size.')
    parser.add_argument('-l', '--num_layers', default=3, type=int,
                        help='GRU layer count. The Aug-GRU implementation is hard-wired '
                             'to 3 layers — see drl_quant.networks.augmented_gru.GRUAug.')
    return parser.parse_args()


def _default_output_path(input_path: str) -> str:
    """``models/rnn/rnn_Quaid_RA-64.dat`` -> ``models/rnn/aug_rnn_Quaid_RA-64.onnx``.

    Matches the convention the C++ training repo's ``aug_rnn_*`` artefacts
    use, and avoids clobbering any existing native-GRU ONNX export sitting
    next to the source ``.dat``.
    """
    p = Path(input_path)
    name = p.name
    stem = name[:-len('.dat')] if name.endswith('.dat') else name[:-len('.pt')] if name.endswith('.pt') else p.stem
    if stem.startswith('rnn_'):
        out_name = 'aug_' + stem + '.onnx'
    else:
        out_name = 'aug_' + stem + '.onnx'
    return str(p.with_name(out_name))


def export(args):
    if not args.output_model:
        args.output_model = _default_output_path(args.input_model)

    scripted_gru = torch.jit.load(args.input_model)
    print('Scripted GRU:'); print(scripted_gru)

    params = {'rnn_hidden_size': args.hidden_size, 'rnn_layers': args.num_layers}
    native_gru = GruNet(args.input_size, params)
    native_gru.load_state_dict(scripted_gru.state_dict())
    native_gru.eval()

    # Transplant nn.GRU weights into the Aug-GRU. Both layouts use a
    # (3*hidden_size, input_size) matrix per layer so the assignment is a
    # straight copy. The Aug-GRU constructor's layer-count check rejects
    # anything outside 1-3.
    gru_aug = GRUAug(args.input_size, args.hidden_size, layers=args.num_layers)
    for i in range(args.num_layers):
        layer = getattr(gru_aug, f'l{i}')
        layer.x2h.weight = getattr(native_gru.rl_gru, f'weight_ih_l{i}')
        layer.h2h.weight = getattr(native_gru.rl_gru, f'weight_hh_l{i}')
    gru_aug.eval()

    # Tracing inputs — the values don't matter, only the shapes.
    dummy_obs = torch.zeros(1, args.input_size, dtype=torch.float32)
    dummy_h = torch.zeros(args.num_layers, 1, args.hidden_size)

    out_dir = os.path.dirname(args.output_model)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            gru_aug,
            (dummy_obs, dummy_h),
            args.output_model,
            verbose=True,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['observations', 'h_t_in'],
            output_names=['features', 'h_t'],
            **LEGACY_EXPORT_KWARGS,
        )
    print(f'Exported {args.output_model}')
    return args.output_model


def main():
    export(get_args())


if __name__ == '__main__':
    main()
