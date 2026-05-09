"""Step 1a: extract a CSV of raw observations from a gzipped training buffer.

The training repo writes per-step buffers as gzipped text where each line is
``obs_0,obs_1,...,obs_N | action,...,action | reward | ...``. For
quantization-calibration purposes we only need the observation slice; this
script keeps the first ``--obs_to_keep`` columns and writes them to a CSV.

Run as::

    python -m drl_quant.data_generation.extract_observations \\
        -i path/to/buffer_<run-id>.csv.gz \\
        -n 17
"""

import argparse
import gzip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input gzipped buffer log (as produced during training).',
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV path. Defaults to <input>_observations_<...>.csv.',
    )
    parser.add_argument(
        '-n', '--obs_to_keep', required=True, type=int,
        help='Number of observation columns to keep (17 for non-RA Quaid, 25 for RA-).',
    )
    args = parser.parse_args()

    out_file = args.output or args.input.replace('buffer_', 'observations_')

    with gzip.open(args.input, 'rt') as fin, open(out_file, 'w') as fout:
        for line in fin:
            obs = line.split('|')[0].split(',')
            fout.write(','.join(obs[0:args.obs_to_keep]))
            fout.write('\n')

    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
