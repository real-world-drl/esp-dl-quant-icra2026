# `drl_quant.networks`

Actor heads and the export-friendly GRU implementation. These are the
`nn.Module`s into which TorchScript checkpoints from the upstream training
repo are loaded before tracing to ONNX.

## Modules

| Module               | Contents                                                                 |
|----------------------|--------------------------------------------------------------------------|
| `td3.py`             | `TD3Actor`, `RTD3Actor` (native `nn.GRU`), `AugRTD3Actor` (Aug-GRU)      |
| `sac.py`             | `DiagGaussianActor`, `RSACActor`, `AugRSACActor`, `SquashedNormal`       |
| `rnn.py`             | `GruNet` — `nn.GRU` wrapper used to load TorchScript GRU weights         |
| `augmented_gru.py`   | `GRUCellAug`, `GRUAug` — the ESP-DL-friendly GRU                         |
| `utils.py`           | `weight_init` (orthogonal Linear init)                                   |

The `*Actor` and `Aug*Actor` pairs share the same `state_dict` shape so
weights from a trained native-GRU actor can be transplanted into the Aug-GRU
variant without re-training.

## Aug-GRU constraints

`augmented_gru.py` is hand-rolled to dodge three ESP-DL ONNX-import bugs.
Editing this module without preserving the constraints **will produce models
that quantize cleanly but output garbage on-device**:

1. **Slice with `tensor.split(size, dim=...)`**, not `chunk` or Python
   slicing. `chunk` exports to a graph ESP-DL handles poorly; slicing
   exports as `Gather` (broken when indexed by a scalar — ESP-DL forces it
   to a tensor and adds a phantom dimension). Tensor-indexed slicing
   exports as `GatherElements` which is unsupported.
2. **Build the new hidden state with `torch.stack`**, never with in-place
   index assignment (`hidden[i] = ...`). The latter exports as `ScatterND`
   which ESP-DL does not support; `stack` becomes `Concat` and is also
   ~1 ms faster on-device.
3. **Reset gate (`i_r`) before update gate (`i_z`)** in the weight layout.
   This matches PyTorch's `nn.GRU` and is the OPPOSITE of Keras. Weights
   copied from a trained `nn.GRU` will silently produce wrong outputs if
   this order is changed.

## Direct use

```python
from drl_quant.networks.td3 import AugRTD3Actor

params = {
    'hidden_size': 256, 'hidden_size2': 256, 'orthogonal_init': False,
    'rnn_hidden_size': 64, 'rnn_layers': 3,
}
model = AugRTD3Actor(state_dim=25, action_dim=8, params=params)
```

In practice you don't construct these yourself — the exporters in
`drl_quant.onnx_export` do it.
