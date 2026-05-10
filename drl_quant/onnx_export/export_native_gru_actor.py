"""Step 2b: export a recurrent actor using PyTorch's **native `nn.GRU`** to ONNX.

This produces an ONNX file containing the standard GRU op. ESP-DL's
quantization path does NOT handle this op robustly, so the resulting model is
useful only as a baseline (it still works fine under
``onnxruntime.quantize_dynamic`` for ONNX-side benchmarking — see
`drl_quant.onnx_dynamic_quantize`). For the ESP-DL deployment path use
`drl_quant.onnx_export.export_aug_actor` instead.

Output path is derived by replacing ``cpp`` -> ``onnx``, ``.dat`` -> ``.onnx``,
and ``act_net`` -> ``with_gru_act_net``.

Run as::

    python -m drl_quant.onnx_export.export_native_gru_actor \\
        -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_RA-TD3_<...>.dat \\
        -s data/QuaidSIM-v4/observations_RA-TD3_<...>.csv \\
        -g models/rnn/rnn_Quaid_RA-64.dat
"""

import argparse
import os

import pandas as pd
import torch

from drl_quant.networks.rnn import GruNet
from drl_quant.networks.sac import DiagGaussianActor, RSACActor
from drl_quant.networks.td3 import RTD3Actor, TD3Actor
from drl_quant.onnx_export._export_compat import LEGACY_EXPORT_KWARGS
from drl_quant.onnx_export._naming import detect_algorithm


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input_model', required=True, help='TorchScript actor .dat.')
    parser.add_argument(
        '-g', '--input_gru_model', default='models/rnn/rnn_Quaid_RA-64.dat',
        help='TorchScript GRU .dat.',
    )
    parser.add_argument('-o', '--output_model', help='Override the output path.')
    parser.add_argument(
        '-n', '--observation_size', default=17, type=int,
        help='Raw observation size: 17 for R-, 25 for RA-.',
    )
    parser.add_argument('-a', '--action_size', default=8, type=int)
    parser.add_argument('-s', '--state_observations', required=True, help='CSV for dummy input.')
    parser.add_argument('-hs1', '--hs1', default=256, type=int)
    parser.add_argument('-hs2', '--hs2', default=256, type=int)
    parser.add_argument('-hs', '--hidden_size', default=64, type=int)
    parser.add_argument('-l', '--num_layers', default=3, type=int)
    return parser.parse_args()


def export(args):
    input_model_path = args.input_model
    if not args.output_model:
        args.output_model = (
            input_model_path
            .replace('cpp', 'onnx')
            .replace('.dat', '.onnx')
            .replace('act_net', 'with_gru_act_net')
        )

    input_size = args.observation_size
    gru_input_size = args.observation_size
    input_size_with_gru = input_size + args.hidden_size
    if 'RA-' in input_model_path:
        input_size_with_gru += args.action_size
        gru_input_size += args.action_size

    scripted_actor = torch.jit.load(input_model_path)
    print('Scripted Actor:'); print(scripted_actor)

    params = {
        'hidden_size': args.hs1,
        'hidden_size2': args.hs2,
        'orthogonal_init': False,
        'rnn_hidden_size': args.hidden_size,
        'rnn_layers': args.num_layers,
    }

    algo = detect_algorithm(input_model_path)
    if algo == 'TD3':
        native_actor = TD3Actor(input_size_with_gru, args.action_size, params)
    else:
        native_actor = DiagGaussianActor(input_size_with_gru, args.action_size, params)
    native_actor.load_state_dict(scripted_actor.state_dict())

    scripted_gru = torch.jit.load(args.input_gru_model)
    print('Scripted GRU:'); print(scripted_gru)

    py_gru = GruNet(gru_input_size, params)
    py_gru.load_state_dict(scripted_gru.state_dict())
    py_gru.eval()

    if algo == 'TD3':
        py_model = RTD3Actor(input_size, args.action_size, params)
        py_model.td3_actor_net = native_actor.td3_actor_net
    else:
        py_model = RSACActor(input_size, args.action_size, params)
        py_model.sac_diag_gaussian_actor = native_actor.sac_diag_gaussian_actor
    py_model.rl_gru = py_gru
    py_model.eval()

    data = pd.read_csv(args.state_observations)
    dummy_obs = torch.tensor(data.tail(1).to_numpy()).float()
    dummy_h = torch.zeros(args.num_layers, 1, args.hidden_size)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            py_model,
            (dummy_obs, dummy_h),
            args.output_model,
            verbose=True,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['observations', 'h_t_in'],
            output_names=['action', 'h_t'],
            **LEGACY_EXPORT_KWARGS,
        )
    print(f'Exported {args.output_model}')
    return args.output_model


def main():
    export(get_args())


if __name__ == '__main__':
    main()
