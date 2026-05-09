"""Step 4a: ESP-DL int8 quantization for **non-recurrent** TD3 / SAC actors.

PPQ's ``espdl_quantize_onnx`` consumes the ONNX file from
`drl_quant.onnx_export.export_actor` plus a calibration ``TensorDataset`` of
observations and emits the on-device-runnable ``.espdl`` bundle. We also
export a NATIVE-format copy so the quantized graph can be re-loaded for
offline numerical comparison without re-quantizing.

Output paths are derived from the input by replacing ``onnx`` -> ``esp-dl``
and ``.onnx`` -> ``.espdl`` (plus sibling ``.native`` / ``.cfg`` files).

Run as::

    python -m drl_quant.espdl_quantize.quantize_actor \\
        -i models/QuaidSIM-v4/onnx/act_net_QuaidSIM-v4_TD3_<...>.onnx \\
        -s data/QuaidSIM-v4/observations_TD3_<...>.pt \\
        -os 17
"""

import argparse

import torch
import ppq.lib as PFL
from ppq import TargetPlatform
from ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader

from drl_quant.constants import test_observations


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input_model', required=True, help='ONNX actor.')
    parser.add_argument('-s', '--calib_dataset', required=True, help='Calibration .pt TensorDataset.')
    parser.add_argument(
        '-os', '--observation_size', type=int, default=17,
        help='Observation size (17 for non-recurrent Quaid).',
    )
    parser.add_argument('-t', '--target', default='esp32s3', help='esp32s3 or esp32p4.')
    parser.add_argument('-nb', '--num_of_bits', default=8, type=int)
    parser.add_argument('-o', '--output_model', help='Override the output .espdl path.')
    return parser.parse_args()


# DEVICE is read by collate_fn; PPQ supplies tensors with a leading batch axis
# we need to strip down to a single sample.
DEVICE = 'cpu'


def collate_fn(batch):
    return batch[0].to(DEVICE)


def main():
    args = get_args()
    onnx_path = args.input_model
    espdl_path = args.output_model or onnx_path.replace('.onnx', '.espdl').replace('onnx', 'esp-dl')

    dataset = torch.load(args.calib_dataset)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    dummy_input = [torch.from_numpy(test_observations).to(DEVICE)]

    quant_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=dataloader,
        calib_steps=100,
        input_shape=[1, args.observation_size],
        inputs=dummy_input,
        target=args.target,
        num_of_bits=args.num_of_bits,
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )

    # Native-format export for offline comparison / debugging.
    PFL.Exporter(TargetPlatform.NATIVE).export(
        file_path=espdl_path.replace('.espdl', '.native'),
        config_path=espdl_path.replace('.espdl', '.cfg'),
        graph=quant_graph,
    )
    print(f'Exported {espdl_path}')


if __name__ == '__main__':
    main()
