# `drl_quant.data_generation` — step 1

Build the calibration data PPQ needs for ESP-DL quantization (step 4).

## Modules

### `extract_observations.py`

Strip a gzipped training-buffer log down to the observation columns and write
a CSV. Only needed if you're starting from raw training output rather than
the bundled CSVs.

```bash
python -m drl_quant.data_generation.extract_observations \
    -i path/to/buffer_<run-id>.csv.gz \
    -n 25                           # 17 for non-RA, 25 for RA-
```

Default output path replaces `buffer_` with `observations_` in the input
filename.

### `generate_calibration.py`

Roll out the (already-exported) ONNX recurrent actor over an observation CSV
and record `(obs, h_t_in, action, h_t_out)` tuples as a PyTorch
`TensorDataset` saved to `.pt`. Required for recurrent ESP-DL quantization
because PPQ has to see realistic GRU hidden-state distributions to calibrate
the GRU activations correctly.

```bash
python -m drl_quant.data_generation.generate_calibration \
    -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv \
    -os 25
```

Default output path replaces `.csv` with `.pt`.

## Notes

- The `.pt` files are gitignored — they are large (~167 MB for the bundled
  RA-TD3 trajectory) and trivially regenerable from the CSVs that ship in
  the repo.
- For the **non-recurrent** TD3 / SAC quantization you still need a `.pt`
  file but it only carries observations + dummy actions; you can build it by
  running this script against any of the non-recurrent ONNX models — only
  the observation column ends up being read by `quantize_actor`.
