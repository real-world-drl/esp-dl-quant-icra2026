"""Step 1b: build a PyTorch ``TensorDataset`` to use as PPQ's calibration set.

For *recurrent* models the calibration set must include the GRU hidden state
``h_t`` at each step, otherwise PPQ has no way to calibrate the GRU
activations. This script runs the (already-exported) ONNX model over the
observation CSV from `extract_observations` and records, for every step, a
``(obs, h_t_in, action, h_t_out)`` tuple. The result is saved as a ``.pt`` file
alongside the input CSV.

Run as::

    python -m drl_quant.data_generation.generate_calibration \\
        -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_<...>.onnx \\
        -s data/QuaidSIM-v4/observations_RA-TD3_<...>.csv \\
        -os 25
"""

import argparse
import time

import numpy as np
import onnxruntime
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '-i', '--input_model', required=True,
        help='ONNX recurrent actor used to roll out the calibration trajectory.',
    )
    parser.add_argument(
        '-s', '--state_observations', required=True,
        help='Observation CSV produced by `extract_observations`.',
    )
    parser.add_argument(
        '-os', '--observation_size', type=int, default=17,
        help='Raw observation size: 17 for R-, 25 for RA- (obs+previous-actions).',
    )
    parser.add_argument('-hs', '--hidden_size', default=64, type=int, help='GRU hidden size.')
    parser.add_argument('-l', '--num_layers', default=3, type=int, help='GRU layer count.')
    parser.add_argument('-o', '--output_dataset_path', help='Defaults to the input CSV with .pt extension.')
    return parser.parse_args()


def generate_dataset(args):
    if not args.output_dataset_path:
        args.output_dataset_path = args.state_observations.replace('csv', 'pt')
        print(f'Output path: {args.output_dataset_path}')

    session = onnxruntime.InferenceSession(args.input_model)
    input_name = session.get_inputs()[0].name
    h_t_name = session.get_inputs()[1].name

    data = pd.read_csv(args.state_observations).to_numpy(dtype=np.float32)
    h_t = np.zeros((args.num_layers, 1, args.hidden_size), dtype=np.float32)

    all_obs, all_actions = [], []
    hidden_states = [h_t]

    total_ms = 0.0
    for row in data:
        start = time.perf_counter()
        obs = np.expand_dims(row[0:args.observation_size], axis=0)
        actions, h_t = session.run([], {input_name: obs, h_t_name: h_t})
        all_obs.append(obs)
        all_actions.append(actions)
        hidden_states.append(h_t)
        total_ms += (time.perf_counter() - start) * 1000.0

    obs_t = torch.tensor(all_obs, dtype=torch.float32)
    h_t_t = torch.tensor(hidden_states[:-1], dtype=torch.float32)
    act_t = torch.tensor(all_actions, dtype=torch.float32)
    out_h_t = torch.tensor(hidden_states[1:], dtype=torch.float32)

    dataset = TensorDataset(obs_t, h_t_t, act_t, out_h_t)
    torch.save(dataset, args.output_dataset_path)
    print(f'Wrote {args.output_dataset_path} ({len(data)} steps, {total_ms:.1f} ms total)')


def main():
    args = get_args()
    generate_dataset(args)


if __name__ == '__main__':
    main()
