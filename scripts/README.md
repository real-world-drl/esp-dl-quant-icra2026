# `scripts/`

End-to-end runners.

## `run_quaidsim_v4.sh`

Regenerates every artefact under `models/QuaidSIM-v4/{onnx,onnx-quant,esp-dl}/`
from the bundled `cpp/*.dat` checkpoints and `data/QuaidSIM-v4/*.csv`
observations. Idempotent — safe to re-run.

```bash
pip install -e .
./scripts/run_quaidsim_v4.sh
```

The script runs steps in this order:

1. **Step 2 first** — ONNX export (we need an ONNX file before we can build
   the recurrent calibration set).
2. **Step 1** — generate `.pt` calibration TensorDatasets by rolling out the
   ONNX models from step 2 over the observation CSVs.
3. **Step 3** — `onnxruntime.quantize_dynamic` over every ONNX in
   `models/QuaidSIM-v4/onnx/`.
4. **Step 4** — PPQ ESP-DL quantization for the non-recurrent and Aug-GRU
   variants. The native-GRU ONNX is intentionally skipped — it does not
   survive ESP-DL's importer.

If you want to run a single step against a single checkpoint, the per-step
invocations are listed in the top-level `README.md` and in each subpackage
README.
