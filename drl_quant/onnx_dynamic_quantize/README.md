# `drl_quant.onnx_dynamic_quantize` — step 3

Apply `onnxruntime.quantize_dynamic` to any of the ONNX exports from step 2.
This is a **host-side benchmark**, not a deployment path: the resulting
`_qd.onnx` runs on a host with `onnxruntime`, not on the MCU. Empirically
this path produces better numerical results than the ESP-DL one — the paper
uses it as the upper-bound int8 reference.

```bash
python -m drl_quant.onnx_dynamic_quantize.quantize \
    -i models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx
```

Output goes to a sibling directory: `.../onnx/<name>.onnx` becomes
`.../onnx-quant/<name>_qd.onnx`. Pass `-o` to override.

This works for all three ONNX flavours from step 2 — non-recurrent,
Aug-GRU, and native-GRU. The `with_gru_act_net_*_qd.onnx` files are useful
specifically because step 4 cannot produce an ESP-DL-deployable counterpart
for the native-GRU graph.
