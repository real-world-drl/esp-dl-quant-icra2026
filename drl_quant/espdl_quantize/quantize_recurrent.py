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
# Espressif forked PPQ into ``esp_ppq`` and moved the ``espdl_*`` entry
# points there. The upstream ``ppq.api.espdl_quantize_onnx`` was removed;
# install ``esp-ppq`` (pulled by our pyproject.toml) and import from the
# fork instead. The function signature is unchanged from the old API.
import esp_ppq.lib as PFL
from esp_ppq import TargetPlatform, QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
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
    parser.add_argument(
        '--error-report', action='store_true',
        help='Enable post-quantization graphwise + layerwise error analysis. Off '
             'by default for recurrent actors because the Aug*Actor.forward path '
             'collapses the batch dim (rank-1 intermediate tensors), which trips '
             "esp-ppq's analyser. The .espdl export is bit-identical with or "
             'without the report.',
    )
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

    # weights_only=False: the calibration .pt is a TensorDataset pickle
    # produced by drl_quant.data_generation.generate_calibration; the new
    # torch >= 2.6 default (weights_only=True) refuses to unpickle anything
    # that isn't a tensor / state_dict. We generate this file ourselves so
    # there's no untrusted-input risk.
    dataset = torch.load(args.calib_dataset, weights_only=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Dummy input is used for graph tracing — shape matters, values don't.
    # We take the first sample from the calibration dataset, which is
    # guaranteed to have the right rank / dim regardless of whether this
    # is a Quaid actor (state_dim 17/25, hidden 64) or a user GRU with
    # arbitrary dims. For byte-stable repro of the bundled QuaidSIM-v4
    # artefacts the deterministic fixtures from drl_quant.constants are
    # still used as a fallback when the dataset format doesn't yield a
    # 2-tuple of (obs, h_t).
    sample = dataset[0]
    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
        dummy_input = [sample[0].to(DEVICE), sample[1].to(DEVICE)]
    else:
        # Older / non-Quaid datasets where each sample is a single tensor —
        # caller probably means to feed the obs only, but we need an h_t
        # too. Fall back to the Quaid fixtures for the legacy path.
        if args.hidden_size == 64:
            h_t_in = test_h_t_in_64
        elif args.hidden_size == 10:
            h_t_in = test_h_t_in_10
        else:
            raise ValueError(
                f'Calibration dataset entries are bare tensors, no h_t_in available, '
                f'and no fixture exists for hidden_size={args.hidden_size}. Either '
                'use a (obs, h_t, ...)-shaped TensorDataset (see '
                'drl_quant.data_generation.generate_calibration) or add an h_t fixture '
                'to drl_quant.constants.'
            )
        dummy_obs = test_observations_with_actions if 'RA-' in onnx_path else test_observations
        dummy_input = [torch.from_numpy(dummy_obs).to(DEVICE),
                       torch.from_numpy(h_t_in).to(DEVICE)]

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
        # Off by default for recurrent actors — see the --error-report flag
        # in get_args() for the full reason. The .espdl is bit-identical
        # either way; only the per-layer noise diagnostic is gated.
        error_report=args.error_report,
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
