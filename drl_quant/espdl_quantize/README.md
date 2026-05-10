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

## `IndexError: Dimension out of range` in graphwise analysis (recurrent only)

After quantization completes (you'll see *"Network Quantization Finished."*)
the analyser may crash with::

    IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
    ... at Op Execution Error: /td3_actor_net/td3_actor_net.0/MatMul

The recurrent actors (`AugRTD3Actor` / `AugRSACActor` and the native-GRU
counterparts) collapse the batch dim in their `forward`:

```python
full_input = torch.cat((rnn_input.squeeze(0).squeeze(0), out.squeeze(0).squeeze(0)))
```

so every Linear in the actor head operates on rank-1 tensors. The new
`esp-ppq`'s `graphwise_error_analyse` calls `tensor.flatten(start_dim=1)`
on every intermediate output, which needs rank ≥ 2. The old `ppq` was
tolerant; the fork is stricter.

`quantize_recurrent.py` defaults `error_report=False` so the analysis is
skipped — **the `.espdl` artefact is bit-identical** either way; you only
lose the per-layer noise diagnostic. Pass `--error-report` to opt back in
once you've refactored the actor to keep the batch dim throughout (e.g.
`torch.cat((rnn_input, out), dim=-1)` and `[..., :action_dim]` indexing in
the SAC head). That refactor changes the ONNX action shape from `(8,)` to
`(1, 8)` so it needs on-device verification before committing.

## `ImportError: cannot import name 'espdl_quantize_onnx' from 'ppq.api'`

Upstream PPQ no longer exposes the ESP-DL entry points. Espressif forked
the project as **esp-ppq**, and the ``espdl_quantize_onnx`` /
``espdl_quantize_torch`` functions now live in ``esp_ppq.api``. Our
quantize scripts already import from the fork; if you set up your env
before this change, install esp-ppq:

```bash
pip install esp-ppq
```

The function signature is unchanged from the original ``ppq.api`` version
so no script edits are needed.

## protobuf 4.x error

If `espdl_quantize_onnx(...)` raises:

```
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date
and must be regenerated with protoc >= 3.19.0.
```

PPQ's compiled extensions use the older protobuf C++ ABI and break
against protobuf 4.x. `pyproject.toml` and `requirements.txt` both pin
`protobuf<4.0` (specifically `protobuf==3.20.3`) so a fresh install
avoids it; if you're seeing this in an existing env, force the
downgrade:

```bash
pip install 'protobuf<4'
```

`onnx 1.16` and `onnxruntime` both work fine on protobuf 3.20.3.

The other workaround the error suggests
(`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) makes parsing 10-100x
slower and isn't worth it for our model sizes — just downgrade.
