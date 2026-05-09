"""Step 4b: ESP-DL int8 quantization for **recurrent (Aug-GRU)** actors.

Same idea as `quantize_actor` but with a two-input graph (observation +
``h_t_in``). Two important details:

* Calibration runs ``batch_size=1`` and the ``collate_fn`` *de-batches* the
  resulting tuple. Aug-GRU's split/stack ops produce wrong slice ordering
  with ``batch_size>1`` during PPQ tracing.
* The matching test fixtures are picked from `drl_quant.constants` based on
  whether the model name contains ``RA-`` (25-dim observation) or not
  (17-dim).

Run as::

    python -m drl_quant.espdl_quantize.quantize_recurrent \\
        -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_<...>.onnx \\
        -s data/QuaidSIM-v4/observations_RA-TD3_<...>.pt \\
        -os 25
"""

import argparse

import torch
import ppq.lib as PFL
from ppq import TargetPlatform, QuantizationSettingFactory
from ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader

from drl_quant.constants import (
    test_h_t_in_10,
    test_h_t_in_64,
    test_observations,
    test_observations_with_actions,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input_model', required=True, help='ONNX recurrent actor (Aug-GRU).')
    parser.add_argument('-s', '--calib_dataset', required=True, help='Calibration .pt TensorDataset.')
    parser.add_argument(
        '-os', '--observation_size', type=int, default=17,
        help='Raw observation size: 17 for R-, 25 for RA-.',
    )
    parser.add_argument('-hs', '--hidden_size', default=64, type=int)
    parser.add_argument('-l', '--num_layers', default=3, type=int)
    parser.add_argument('-t', '--target', default='esp32s3', help='esp32s3 or esp32p4.')
    parser.add_argument('-nb', '--num_of_bits', default=8, type=int)
    parser.add_argument('-o', '--output_model', help='Override the output .espdl path.')
    return parser.parse_args()


DEVICE = 'cpu'


def collate_fn(batch):
    # batch_size=1 -> de-batch obs + h_t separately so PPQ traces with shape
    # (1, obs_size) and (num_layers, 1, hidden_size). Larger batches break
    # Aug-GRU's split ordering during quantization.
    return [batch[0].to(DEVICE).squeeze(0), batch[1].to(DEVICE).squeeze(0)]


def main():
    args = get_args()
    onnx_path = args.input_model
    espdl_path = args.output_model or onnx_path.replace('.onnx', '.espdl').replace('onnx', 'esp-dl')

    dataset = torch.load(args.calib_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if args.hidden_size == 64:
        h_t_in = test_h_t_in_64
    elif args.hidden_size == 10:
        h_t_in = test_h_t_in_10
    else:
        raise ValueError(f'No fixed h_t_in for hidden_size={args.hidden_size}; add one to drl_quant.constants.')

    dummy_obs = test_observations_with_actions if 'RA-' in onnx_path else test_observations
    dummy_input = [torch.from_numpy(dummy_obs).to(DEVICE), torch.from_numpy(h_t_in).to(DEVICE)]

    setting = QuantizationSettingFactory.espdl_setting()

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=dataloader,
        calib_steps=1000,
        input_shape=None,
        inputs=dummy_input,
        target=args.target,
        num_of_bits=args.num_of_bits,
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        setting=setting,
        verbose=1,
    )

    PFL.Exporter(TargetPlatform.NATIVE).export(
        file_path=espdl_path.replace('.espdl', '.native'),
        config_path=espdl_path.replace('.espdl', '.cfg'),
        graph=quant_graph,
    )
    print(f'Exported {espdl_path}')


if __name__ == '__main__':
    main()
