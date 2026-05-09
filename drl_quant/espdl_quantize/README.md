# `drl_quant.espdl_quantize` — step 4

ONNX -> ESP-DL int8 via PPQ's `espdl_quantize_onnx`. Two modules because the
non-recurrent and recurrent paths have different input shapes and different
calibration-loader semantics.

Each call produces three files alongside the input ONNX:

- `<name>.espdl` — the on-device-runnable artefact, flash this to the MCU
- `<name>.native` + `<name>.cfg` — PPQ NATIVE format, used to re-load the
  quantized graph host-side (e.g. for numerical comparison without
  re-quantizing)

## Modules

### `quantize_actor.py` — non-recurrent

For `act_net_*` ONNX models (TD3 / SAC, no GRU). Simpler: single observation
input, larger calibration batches (1000), fewer steps (100).

```bash
python -m drl_quant.espdl_quantize.quantize_actor \
    -i models/QuaidSIM-v4/onnx/act_net_QuaidSIM-v4_TD3_+225.827_750000.onnx \
    -s data/QuaidSIM-v4/observations_TD3_2025-09-08T22-45-15.pt \
    -os 17
```

### `quantize_recurrent.py` — recurrent (Aug-GRU)

For `aug_act_net_*` ONNX models. Two-input graph (observation + `h_t_in`),
batch size **must be 1**, more calibration steps (1000):

```bash
python -m drl_quant.espdl_quantize.quantize_recurrent \
    -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.pt \
    -os 25
```

The collator de-batches a batch-of-1 because Aug-GRU's split/stack ops
produce wrong slice ordering at PPQ trace time when `batch_size > 1`.

The dummy input passed to PPQ is picked from `drl_quant.constants` based on
the filename: `RA-` in the path -> 25-dim observation, otherwise 17-dim.
This *must* match the ONNX input shape.

## Targets

`-t esp32s3` (default) or `-t esp32p4`. The bundled `.espdl` files in
`models/QuaidSIM-v4/esp-dl/` were produced for `esp32s3`.

## Trying to quantize a `with_gru_act_net_*` model

Don't — the native-GRU op trips up ESP-DL's importer. Use
`drl_quant.onnx_export.export_aug_actor` instead and quantize the resulting
`aug_act_net_*` ONNX.
