"""Step 3: dynamic int8 quantization of an ONNX model via onnxruntime.

This is a *non-deployment* path used for benchmarking only: the resulting
``_qd.onnx`` runs on a host with ``onnxruntime``, not on the MCU. Empirically
it produces better numerical results than the ESP-DL path (the paper uses it
as the upper-bound reference for the int8 budget).

Output goes to a sibling directory: ``models/.../onnx/`` -> ``onnx-quant/``,
suffixed with ``_qd.onnx``.

Run as::

    python -m drl_quant.onnx_dynamic_quantize.quantize \\
        -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_<...>.onnx
"""

import argparse
import os

from onnxruntime.quantization import quantize_dynamic


def get_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-i', '--input_model', required=True, help='Input ONNX model.')
    parser.add_argument('-o', '--output_model', help='Override the output path (defaults to onnx-quant/<name>_qd.onnx).')
    return parser.parse_args()


def main():
    args = get_args()
    if args.output_model is None:
        args.output_model = args.input_model.replace('onnx/', 'onnx-quant/').replace('.onnx', '_qd.onnx')

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    quantize_dynamic(args.input_model, args.output_model)
    print(f'Quantized model saved to {args.output_model}')


if __name__ == '__main__':
    main()
