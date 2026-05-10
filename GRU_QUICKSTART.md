# Standalone GRU quantization for ESP-DL — quickstart

For when you have a **PyTorch-trained `nn.GRU`** (language-model
preprocessor, sequence encoder, etc.) and want to deploy it quantized on
an ESP32-S3 / ESP32-P4 via ESP-DL. This walkthrough skips the DRL-specific
parts of the rest of the repo — you don't need a Quaid robot, an MQTT
broker, or any of the actor / policy machinery.

End-to-end you'll get `your_gru.pt` (PyTorch) -> `aug_your_gru.onnx`
(intermediate ONNX) -> `aug_your_gru.espdl` (quantized, ready to flash).

## Why this exists

ESP-DL's ONNX importer doesn't handle the standard `GRU` op robustly, so
the obvious `torch.onnx.export(your_gru, ...)` produces a model that won't
run on-device. This repo's *Aug-GRU* is a hand-rolled rebuild of `nn.GRU`
from primitives that ESP-DL does support. We transplant your trained
weights into the Aug-GRU and quantize *that*. Numerical equivalence with
the original `nn.GRU` is verified by tests (see
`drl_quant/networks/tests/test_augmented_gru.py`).

## What's supported

| Property              | Range                                           |
|-----------------------|-------------------------------------------------|
| GRU layers            | 1, 2, or 3                                      |
| Bias                  | yes (default `nn.GRU` `bias=True`) or no        |
| Input dim             | any                                             |
| Hidden size           | any                                             |
| Bidirectional         | not supported (Aug-GRU is unidirectional)       |
| Wrapper module        | OK — auto-detected (e.g. `model.encoder.gru.*`) |

If your GRU has more than 3 layers, you'll need to extend
`drl_quant.networks.augmented_gru.GRUAug` to add `self.l3` etc. and
matching branches in `forward()`. 1-3 covers what we've seen in practice.

## Setup

```bash
git clone https://github.com/<...>/esp-dl-quant-icra2026
cd esp-dl-quant-icra2026
pip install -e .
```

PyTorch ≥ 2.5 and `esp-ppq` are pulled in automatically. See the main
`README.md`'s **Setup** section if any step trips on protobuf / torch
version issues.

## Step 1 — save your trained GRU

After training, write the state dict (or the full module — both work):

```python
import torch
torch.save(model.state_dict(), 'my_gru.pt')
```

The exporter accepts:

* a bare `nn.GRU.state_dict()` (`torch.save(model.state_dict(), ...)`)
* a wrapped `nn.Module.state_dict()` where the GRU is at any depth
  (e.g. `encoder.rnn.weight_ih_l0`) — the prefix is stripped automatically
* a full module pickle (`torch.save(model, ...)`)
* a TorchScript file (`torch.jit.save(...)`)
* the C++ libtorch `torch::save(...)` archive format (`.dat`)

There's no need to convert between these formats — the exporter
auto-detects.

## Step 2 — generate a calibration dataset

Quantization needs a small representative sample of inputs to calibrate
activation ranges. For GRUs the calibration set is a list of
`(observation, h_t_in)` tuples:

```python
import torch
from torch.utils.data import TensorDataset

# Stack ~1000 representative samples from your inference distribution.
# Shapes (matching what nn.GRU expects):
#   observations: (1, input_dim)         — one sequence step at a time
#   hidden state: (num_layers, 1, hidden_size)
n_steps = 1000
obs   = torch.randn(n_steps, 1, INPUT_DIM)                   # ← replace with real data
h_t   = torch.randn(n_steps, NUM_LAYERS, 1, HIDDEN_SIZE)     # ← replace with real data

dataset = TensorDataset(obs, h_t)
torch.save(dataset, 'calib.pt')
```

You almost certainly want **real** observations and hidden states from
running your trained model on validation data, not random tensors —
calibration with random inputs gives you a quantizer tuned for noise.
Quickest way: run your model on a held-out batch and capture
`(input, h_t_at_each_step)` tuples.

## Step 3 — export to Aug-GRU ONNX

The exporter auto-detects everything from the checkpoint:

```bash
python -m drl_quant.onnx_export.export_aug_gru -i my_gru.pt
# -> writes aug_my_gru.onnx alongside my_gru.pt
```

Want to override autodetection (e.g. for a sanity check)?

```bash
python -m drl_quant.onnx_export.export_aug_gru \
    -i my_gru.pt \
    -hs 128 -l 2 --bias \
    -o /tmp/aug_my_gru.onnx
```

The output ONNX has two inputs (`observations`, `h_t_in`) and two outputs
(`features`, `h_t`).

You can sanity-check it against the original PyTorch GRU before
quantizing:

```python
import numpy as np, onnxruntime as ort, torch
sess = ort.InferenceSession('aug_my_gru.onnx')

x = torch.randn(1, 1, INPUT_DIM)
h = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE)

ref_out, ref_h = your_trained_gru(x, h)
out, h_new = sess.run(['features', 'h_t'],
                      {'observations': x.squeeze(1).numpy().astype(np.float32),
                       'h_t_in': h.numpy()})

assert np.allclose(ref_h.numpy(), h_new, atol=1e-5)  # exact match expected
```

## Step 4 — quantize for ESP-DL

```bash
python -m drl_quant.espdl_quantize.quantize_recurrent \
    -i aug_my_gru.onnx \
    -s calib.pt \
    -t esp32s3                  # or esp32p4
# -> writes aug_my_gru.espdl + aug_my_gru.native + aug_my_gru.cfg
```

The `-os` (observation size) flag exists for the Quaid pipeline but isn't
required for plain GRU use — the exporter reads dims from the ONNX. If
the calibration loader complains about hidden size, pass `-hs` and `-l`
to match your GRU.

`-t` selects the target chip. `esp32s3` and `esp32p4` are supported; pick
whichever you're flashing.

The `.espdl` file is what you flash. The `.native` + `.cfg` pair is
PPQ's native format — useful if you want to re-load the quantized graph
host-side for a numerical comparison without re-quantizing.

## Step 5 — use the .espdl on-device

ESP-DL's official Python-quantized-model tutorial covers loading the
`.espdl` from your ESP-IDF firmware and feeding it data:
https://docs.espressif.com/projects/esp-dl/en/latest/

The model takes two inputs (`observations` and `h_t_in`) and produces two
outputs (`features` and `h_t`). On-device, you'll typically:

1. Allocate a persistent buffer for `h_t` (the GRU's recurrent state),
   initialised to zeros.
2. For each new input, copy it to the model's `observations` input
   tensor and copy your retained `h_t` into `h_t_in`.
3. Run inference, then copy the output `h_t` back into your buffer for
   the next call.

That's the same recurrence pattern any ONNX-runtime-backed inference loop
uses — see `drl_quant.inference.preprocessors.OnnxGruPreprocessor` in
this repo for a host-side reference implementation.

## Common issues

**`No GRU weights found in checkpoint`**
The exporter looks for keys ending in `weight_ih_l0`. If your model
stores the GRU somewhere unusual (e.g. under a non-standard layer name),
save the GRU submodule on its own first:
`torch.save(model.gru.state_dict(), 'gru.pt')`.

**`Multiple GRUs found in the checkpoint, cannot disambiguate`**
Same fix — save the specific GRU submodule on its own.

**`GRUAug supports (1, 2, 3) layers, got 4`**
Aug-GRU is intentionally hard-wired to ≤ 3 layers (each layer is a named
module so ONNX traces a fixed graph). Either retrain with
`num_layers ≤ 3` or extend `drl_quant.networks.augmented_gru.GRUAug` to
add `self.l3`.

**Bias mismatch when loading**
The exporter auto-detects `bias` from the checkpoint, but if you trained
with a non-standard layout the `--bias` / `--no-bias` flags force it.

**Quantized output diverges from ONNX output**
Expected — quantization is lossy. Compare against
`drl_quant/onnx_dynamic_quantize/quantize.py` (host-side `int8`
benchmark) for the closest "ideal" int8 reference. If the gap is much
larger than that, your calibration set probably isn't representative.
Re-generate with real samples from your inference distribution.

## Citing this in a paper

The Aug-GRU rebuild and the ESP-DL deployment recipe are the
contributions of *Quantization of DRL Models for Embedded Microcontrollers*
(ICRA 2026). Citation in the main `README.md` once the paper is published.
