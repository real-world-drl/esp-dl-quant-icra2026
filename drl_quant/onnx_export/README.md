# `drl_quant.onnx_export` — step 2

TorchScript `.dat` -> ONNX. Four modules:

* three actor exporters (one per GRU strategy: none / Aug-GRU baked in /
  native `nn.GRU` baked in);
* one **GRU-only** exporter for use cases where you don't have an actor
  head — language-model preprocessing, generic sequence encoders, or
  composing the quantized GRU with an external actor.

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

### `export_aug_gru.py` — **GRU-only** (no actor head)

For use cases where you want a quantized GRU on its own — language-model
preprocessing, generic sequence encoders, or composing the GRU with an
external actor head. Loads a TorchScript `nn.GRU` checkpoint, transplants
the weights into a fresh `GRUAug`, and writes a two-input / two-output
ONNX (`observations` + `h_t_in` -> `features` + `h_t`). Output filename
follows the `aug_` prefix convention used by the C++ training repo:
`rnn_<...>.dat` -> `aug_rnn_<...>.onnx`.

```bash
python -m drl_quant.onnx_export.export_aug_gru \
    -i models/rnn/rnn_Quaid_RA-64.dat \
    -n 25                                # GRU input dim: 17 for R-, 25 for RA-
```

Why Aug-GRU and not the standard ONNX GRU op? Same reason as
`export_aug_actor` above — ESP-DL's importer cannot handle the standard
GRU op robustly. The Aug-GRU rebuild uses only the primitives ESP-DL is
happy with. See `drl_quant/networks/README.md` for the full constraint
list.

The resulting ONNX feeds straight into:

* **`drl_quant.onnx_dynamic_quantize.quantize`** — host-side int8 benchmark.
* **`drl_quant.espdl_quantize.quantize_recurrent`** — ESP-DL int8
  deployment bundle. The same calibration loader works because the input
  shape (`obs` + `h_t`) is identical to the actor-with-GRU case.
* **`drl_quant.inference.preprocessors.OnnxGruPreprocessor`** — use the
  quantized GRU as a feature extractor in front of a separate actor.

`GRUAug` supports 1, 2 or 3 layers — pass `--num_layers` (default 3) to
match your trained checkpoint. 1 covers language-model / sequence-encoder
use cases; 2-3 cover the QuaidSIM-v4 trained policies. The exporter
delegates the depth check to `GRUAug`'s constructor, so unsupported counts
(e.g. 4) fail before any file IO with a clear error.

## Inputs picked from the filename

All three exporters branch on substrings in the input path:

- `TD3` or `SAC` (required) -> selects `TD3Actor` (Tanh, action_dim outputs)
  vs `DiagGaussianActor` (Gaussian, action_dim*2 outputs). Filenames
  missing both tags raise a `ValueError` with examples — the heads are not
  interchangeable, so silent fallback would crash later inside
  `load_state_dict`.
- `RA-` -> add 8 (action-size) to the GRU input dimension; observation
  size becomes 25 instead of 17.

Keep these conventions in any new checkpoint filenames.

## torch >= 2.5 notes

These exporters target the legacy TorchScript-based `torch.onnx.export`
path (the one that traces `nn.Module`s and emits a static graph). Since
torch 2.5 there's also a new dynamo-based exporter, and torch 2.6+ has
been gradually shifting `torch.onnx.export` toward it. The dynamo path:

* fails with `AttributeError: 'LeafSpec' object has no attribute 'type'`
  on the `Aug*Actor` graph shapes;
* refuses any `opset_version` below 18.

To stay on the legacy path, every `torch.onnx.export(...)` call here
spreads `**LEGACY_EXPORT_KWARGS` from `_export_compat.py`, which evaluates
to `{'dynamo': False}` on torch >= 2.5 and `{}` on older torches (where
the parameter doesn't exist). If a future torch removes the legacy
exporter entirely, this is the one place to update.

Required deps for the new-torch path:

```bash
pip install onnxscript    # torch >= 2.5 imports it even on the legacy path
```

`onnxscript` is now a declared dependency in `pyproject.toml` so a fresh
`pip install -e .` picks it up automatically.
