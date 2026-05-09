# `drl_quant.onnx_export` — step 2

TorchScript `.dat` -> ONNX. Three modules because the GRU has to be handled
three different ways.

## Modules

### `export_actor.py` — non-recurrent

For `act_net_*_TD3_*.dat` and `act_net_*_SAC_*.dat`. Single-input ONNX with
just the observation tensor.

```bash
python -m drl_quant.onnx_export.export_actor \
    -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_TD3_+225.827_750000.dat \
    -s data/QuaidSIM-v4/observations_TD3_2025-09-08T22-45-15.csv
```

### `export_aug_actor.py` — recurrent (Aug-GRU) — **deployment path**

For `act_net_*_R-*.dat` and `act_net_*_RA-*.dat`. Loads both the actor and
the GRU TorchScript checkpoints, transplants the GRU weights into a fresh
`GRUAug`, wraps everything into the matching `Aug*Actor` and traces. Output
filename gets the `aug_act_net` prefix.

```bash
python -m drl_quant.onnx_export.export_aug_actor \
    -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv \
    -g models/rnn/rnn_Quaid_RA-64.dat
```

This is the only recurrent path that survives ESP-DL quantization; see
`drl_quant/networks/README.md` for the constraints that make Aug-GRU
ESP-DL-safe.

### `export_native_gru_actor.py` — recurrent (native `nn.GRU`)

Same as above but keeps PyTorch's `nn.GRU`, producing an ONNX with the
standard GRU op. ESP-DL's import does not handle this op robustly; the
output is useful only as a baseline (and as input to step 3, where
`onnxruntime.quantize_dynamic` handles it fine). Output filename gets the
`with_gru_act_net` prefix.

```bash
python -m drl_quant.onnx_export.export_native_gru_actor \
    -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv \
    -g models/rnn/rnn_Quaid_RA-64.dat
```

## Inputs picked from the filename

All three exporters branch on substrings in the input path:

- `TD3` -> use `TD3Actor` head, otherwise `DiagGaussianActor` (SAC head)
- `RA-` -> add 8 (action-size) to the GRU input dimension; observation
  size becomes 25 instead of 17

Keep these conventions in any new checkpoint filenames.
