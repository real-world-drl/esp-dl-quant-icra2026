# DRL-Quant — ICRA 2026 companion

Quantization toolchain that converts trained Quaid quadruped DRL actor
networks into [ESP-DL](https://github.com/espressif/esp-dl) `.espdl` models
for on-device inference on ESP32-S3 / ESP32-P4. Submitted alongside the ICRA
2026 paper *Quantization of DRL Models for Embedded Microcontrollers* as one
of three companion repositories; the paper itself and the experimental
methodology live in the main repo. This repo is the model side: TorchScript
in, `.espdl` out.

The shipped pipeline targets the **QuaidSIM-v4** environment and includes the
six trained TorchScript actors (TD3, SAC, RA-TD3, RA-SAC) plus the calibration
observations needed to reproduce every step end-to-end.

## Repository layout

```
data/QuaidSIM-v4/              raw observation CSVs (calibration source)
models/rnn/                    shared GRU weights transplanted into Aug-GRU
models/QuaidSIM-v4/
    cpp/                       TorchScript .dat actors (inputs)
    onnx/                      ONNX exports (intermediate)
    onnx-quant/                onnxruntime-dynamic-quantized ONNX (benchmark)
    esp-dl/                    ESP-DL .espdl bundles (deployment)
drl_quant/                     Python package
    networks/                  Aug-GRU + actor heads
    data_generation/           step 1: build calibration set
    onnx_export/               step 2: TorchScript -> ONNX (3 variants)
    onnx_dynamic_quantize/     step 3: ONNX -> dynamic-quantized ONNX
    espdl_quantize/            step 4: ONNX -> ESP-DL .espdl
    inference/                 evaluation Player + runners + preprocessors
quaid_env/                     standalone gymnasium env (own pyproject.toml,
                               separate install — see quaid_env/README.md)
scripts/                       end-to-end runner
```

Each subpackage has its own `README.md` explaining what it does and how to
invoke it directly.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or, for byte-stable reproduction of the bundled artefacts:
pip install -r requirements.txt && pip install -e . --no-deps
```

PPQ ships with C extensions; on a fresh box you may need
``pip install --upgrade pip setuptools wheel`` first.

## End-to-end pipeline

The four steps below mirror what `scripts/run_quaidsim_v4.sh` does. Every
script supports `-h` for the full argument list. All commands assume you run
from the repo root.

### Step 1 — calibration data

`data/QuaidSIM-v4/*.csv` are already shipped. To rebuild the calibration
`.pt` from a CSV (recurrent variants only):

```bash
python -m drl_quant.data_generation.generate_calibration \
    -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv \
    -os 25
```

If you want to start one step earlier, from a gzipped training-buffer log:

```bash
python -m drl_quant.data_generation.extract_observations \
    -i path/to/buffer_<run-id>.csv.gz -n 25
```

`.pt` files are gitignored — regenerate them on first run.

### Step 2 — ONNX export (three variants)

Pick the one that matches the actor:

```bash
# Non-recurrent TD3 / SAC
python -m drl_quant.onnx_export.export_actor \
    -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_TD3_+225.827_750000.dat \
    -s data/QuaidSIM-v4/observations_TD3_2025-09-08T22-45-15.csv

# Recurrent + Aug-GRU (the only variant that survives ESP-DL quantization)
python -m drl_quant.onnx_export.export_aug_actor \
    -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv \
    -g models/rnn/rnn_Quaid_RA-64.dat

# Recurrent + native nn.GRU (baseline; not deployable to ESP-DL)
python -m drl_quant.onnx_export.export_native_gru_actor \
    -i models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.csv \
    -g models/rnn/rnn_Quaid_RA-64.dat
```

### Step 3 — dynamic ONNX quantization (host-side benchmark)

```bash
python -m drl_quant.onnx_dynamic_quantize.quantize \
    -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx
```

This is the upper-bound int8 reference used in the paper. Output goes to
`models/.../onnx-quant/<name>_qd.onnx`.

### Step 4 — ESP-DL quantization (deployment)

```bash
# Non-recurrent
python -m drl_quant.espdl_quantize.quantize_actor \
    -i models/QuaidSIM-v4/onnx/act_net_QuaidSIM-v4_TD3_+225.827_750000.onnx \
    -s data/QuaidSIM-v4/observations_TD3_2025-09-08T22-45-15.pt \
    -os 17

# Recurrent (Aug-GRU)
python -m drl_quant.espdl_quantize.quantize_recurrent \
    -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    -s data/QuaidSIM-v4/observations_RA-TD3_2025-09-08T22-45-15.pt \
    -os 25
```

Each invocation produces a `.espdl` (the on-device artefact) plus a
`.native` + `.cfg` pair (PPQ NATIVE format, useful for offline numerical
comparison). Flash the `.espdl` per the ESP-DL deployment notes in the
paper's main repo.

### Run all four steps

```bash
./scripts/run_quaidsim_v4.sh
```

The script is non-destructive — re-running it overwrites `models/QuaidSIM-v4/onnx/`,
`onnx-quant/`, and `esp-dl/` with freshly generated copies of the bundled
artefacts.

### Evaluation / inference

`drl_quant.inference` runs a quantized actor against the QuaidEnv (over
MQTT) and records inference-time + reward statistics. The runner +
observation-preprocessor are auto-detected from the model filename (TD3 /
SAC / R-/RA-/A- variants, TorchScript or ONNX, Aug-GRU baked in or external).

```bash
pip install -e quaid_env/
python -m drl_quant.inference \
    --model models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    --env-config quaid_env/examples/quaid-icra-sim.yaml \
    --episodes 5 --output-dir runs/eval-1
```

See `drl_quant/inference/README.md` for the full dispatch table and the
list of supported model formats.

## Naming conventions (load-bearing)

Several scripts branch on substrings in the input filename:

| Prefix / marker          | Meaning                                                    |
|--------------------------|------------------------------------------------------------|
| `act_net_`               | non-recurrent actor (TD3 / SAC), 17-dim observation        |
| `aug_act_net_`           | Aug-GRU recurrent actor, exported by `export_aug_actor`    |
| `with_gru_act_net_`      | native `nn.GRU` recurrent actor, baseline only             |
| `R-TD3` / `R-SAC`        | recurrent (observation-only into GRU)                      |
| `RA-TD3` / `RA-SAC`      | recurrent + previous actions; observation is 25-dim        |

Keep these in any new filenames or step 4 will pick the wrong fixture from
`drl_quant/constants.py`.

## License

MIT — see `LICENSE`.
