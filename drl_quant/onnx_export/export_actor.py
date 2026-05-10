"""Step 2a: export a *non-recurrent* TD3 / SAC actor to ONNX.

Loads a TorchScript ``.dat`` checkpoint, transplants the weights into the
matching `nn.Module` from `drl_quant.networks`, and writes an ONNX file with a
single observation input and a single action output.

The actor type (TD3 vs SAC) is inferred from the ``TD3`` substring in the
filename. Output path defaults to the input path with ``cpp`` -> ``onnx`` and
``.dat`` -> ``.onnx``.

Run as::

    python -m drl_quant.onnx_export.export_actor \\
        -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_TD3_<...>.dat \\
        -s data/QuaidSIM-v4/observations_TD3_<...>.csv
"""

import argparse
import os

import pandas as pd
import torch

from drl_quant.networks.sac import DiagGaussianActor
from drl_quant.networks.td3 import TD3Actor
from drl_quant.onnx_export._export_compat import LEGACY_EXPORT_KWARGS
from drl_quant.onnx_export._naming import detect_algorithm


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input_model', required=True, help='TorchScript .dat actor.')
    parser.add_argument('-o', '--output_model', help='Override the output path.')
    parser.add_argument(
        '-n', '--observation_size', default=17, type=int,
        help='Observation size (17 for non-recurrent Quaid).',
    )
    parser.add_argument('-a', '--action_size', default=8, type=int, help='Action size (8 for Quaid).')
    parser.add_argument(
        '-s', '--state_observations', required=True,
        help='CSV used to source a single dummy input row for tracing.',
    )
    return parser.parse_args()


def export(args):
    input_model_path = args.input_model
    output_model_path = args.output_model or input_model_path.replace('cpp', 'onnx').replace('.dat', '.onnx')

    scripted_model = torch.jit.load(input_model_path)
    print('Scripted Model:'); print(scripted_model)

    params = {'hidden_size': 256, 'hidden_size2': 256, 'orthogonal_init': False}

    if detect_algorithm(input_model_path) == 'TD3':
        py_model = TD3Actor(args.observation_size, args.action_size, params)
    else:
        py_model = DiagGaussianActor(args.observation_size, args.action_size, params)

    print(py_model)
    py_model.load_state_dict(scripted_model.state_dict())
    py_model.eval()

    data = pd.read_csv(args.state_observations)
    dummy_input = torch.tensor(data.tail(1).to_numpy()).float()

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.onnx.export(
        py_model,
        dummy_input,
        output_model_path,
        verbose=True,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['observations'],
        output_names=['action'],
        **LEGACY_EXPORT_KWARGS,
    )
    print(f'Exported {output_model_path}')
    return output_model_path


def main():
    export(get_args())


if __name__ == '__main__':
    main()
