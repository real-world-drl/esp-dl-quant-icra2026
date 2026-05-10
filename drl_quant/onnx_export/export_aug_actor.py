"""Step 2c: export a recurrent actor using **Aug-GRU** to ONNX.

This is the path that survives ESP-DL quantization. We:

1. Load the TorchScript actor checkpoint into the matching native-GRU
   `nn.Module` (`RTD3Actor` / `RSACActor`).
2. Load the TorchScript GRU checkpoint into a `GruNet`.
3. Build a fresh `GRUAug` with the same shape and copy weights across:
   ``weight_ih_l<N>`` -> ``l<N>.x2h.weight`` and ``weight_hh_l<N>`` ->
   ``l<N>.h2h.weight`` (no bias — the upstream `nn.GRU` was constructed with
   ``bias=False``).
4. Wrap the head + Aug-GRU into the matching `Aug*Actor` and trace to ONNX.

The actor type is inferred from ``TD3`` / ``RA-`` substrings in the input
filename. Output path is derived by replacing ``cpp`` -> ``onnx``,
``.dat`` -> ``.onnx``, and ``act_net`` -> ``aug_act_net``.

Run as::

    python -m drl_quant.onnx_export.export_aug_actor \\
        -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_RA-TD3_<...>.dat \\
        -s data/QuaidSIM-v4/observations_RA-TD3_<...>.csv \\
        -g models/rnn/rnn_Quaid_RA-64.dat
"""

import argparse
import os

import pandas as pd
import torch

from drl_quant.networks.augmented_gru import GRUAug
from drl_quant.networks.rnn import GruNet
from drl_quant.networks.sac import AugRSACActor, DiagGaussianActor
from drl_quant.networks.td3 import AugRTD3Actor, TD3Actor
from drl_quant.onnx_export._export_compat import LEGACY_EXPORT_KWARGS
from drl_quant.onnx_export._naming import detect_algorithm


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input_model', required=True, help='TorchScript actor .dat.')
    parser.add_argument(
        '-g', '--input_gru_model', default='models/rnn/rnn_Quaid_RA-64.dat',
        help='TorchScript GRU .dat (weights to be transplanted into Aug-GRU).',
    )
    parser.add_argument('-o', '--output_model', help='Override the output path.')
    parser.add_argument(
        '-n', '--observation_size', default=17, type=int,
        help='Raw observation size: 17 for R-, 25 for RA-.',
    )
    parser.add_argument('-a', '--action_size', default=8, type=int, help='Action size.')
    parser.add_argument('-s', '--state_observations', required=True, help='CSV for dummy input.')
    parser.add_argument('-hs1', '--hs1', default=256, type=int)
    parser.add_argument('-hs2', '--hs2', default=256, type=int)
    parser.add_argument('-hs', '--hidden_size', default=64, type=int, help='GRU hidden size.')
    parser.add_argument('-l', '--num_layers', default=3, type=int, help='GRU layer count.')
    return parser.parse_args()


def export(args):
    input_model_path = args.input_model
    if not args.output_model:
        args.output_model = (
            input_model_path
            .replace('cpp', 'onnx')
            .replace('.dat', '.onnx')
            .replace('act_net', 'aug_act_net')
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

    native_gru = GruNet(gru_input_size, params)
    native_gru.load_state_dict(scripted_gru.state_dict())
    native_gru.eval()

    # Transplant nn.GRU weights into the Aug-GRU.
    gru_aug = GRUAug(gru_input_size, params['rnn_hidden_size'])
    gru_aug.l0.x2h.weight = native_gru.rl_gru.weight_ih_l0
    gru_aug.l0.h2h.weight = native_gru.rl_gru.weight_hh_l0
    gru_aug.l1.x2h.weight = native_gru.rl_gru.weight_ih_l1
    gru_aug.l1.h2h.weight = native_gru.rl_gru.weight_hh_l1
    gru_aug.l2.x2h.weight = native_gru.rl_gru.weight_ih_l2
    gru_aug.l2.h2h.weight = native_gru.rl_gru.weight_hh_l2

    if algo == 'TD3':
        aug_actor = AugRTD3Actor(input_size, args.action_size, params)
        aug_actor.td3_actor_net = native_actor.td3_actor_net
    else:
        aug_actor = AugRSACActor(input_size, args.action_size, params)
        aug_actor.sac_diag_gaussian_actor = native_actor.sac_diag_gaussian_actor
    aug_actor.rl_gru = gru_aug
    aug_actor.eval()

    data = pd.read_csv(args.state_observations)
    dummy_obs = torch.tensor(data.tail(1).to_numpy()).float()
    dummy_h = torch.zeros(args.num_layers, 1, args.hidden_size)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            aug_actor,
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
