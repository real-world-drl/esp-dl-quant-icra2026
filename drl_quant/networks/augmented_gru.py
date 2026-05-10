"""Aug-GRU: a hand-rolled GRUCell / multi-layer GRU built from primitives that
ESP-DL's ONNX importer handles correctly.

Why this exists
---------------
PyTorch's `nn.GRU` exports as the ONNX `GRU` op. ESP-DL only partially supports
that op, and several "obvious" reimplementations break in subtle ways on the
device. The constraints below are load-bearing — please preserve them when
editing this file:

* Use ``tensor.split(size, dim=...)`` to slice the gate tensors. ``chunk``
  exports to a graph ESP-DL handles poorly; Python slicing exports as ``Gather``
  which ESP-DL forces to a tensor and adds a phantom dimension. Tensor-indexed
  slicing exports as ``GatherElements`` which is unsupported.
* Build the new hidden state with ``torch.stack``, never with in-place index
  assignment (``hidden[i] = ...``). The latter exports as ``ScatterND`` which
  ESP-DL does not support; ``stack`` becomes ``Concat`` and is also ~1 ms
  faster on-device.
* The reset gate (``i_r``) comes BEFORE the update gate (``i_z``) in the
  weight layout. This matches PyTorch's `nn.GRU` and is the OPPOSITE of Keras.
  Weights copied from a trained `nn.GRU` will silently produce wrong outputs
  if this order is changed.

Numerical equivalence with ``torch.nn.GRU``
-------------------------------------------
``GRUAug(layers=N, bias=False)`` produces the same outputs as
``torch.nn.GRU(num_layers=N, bias=False)`` once weights are transplanted
layer-by-layer (``weight_ih_l<N>`` -> ``l<N>.x2h.weight``,
``weight_hh_l<N>`` -> ``l<N>.h2h.weight``). This is verified in
``tests/test_augmented_gru.py`` for layer counts 1, 2 and 3.

(Earlier revisions of this module had a multi-layer cascade bug where layer
N+1's input was the *previous* timestep's layer-N hidden state instead of
the *current* layer-N output. The wrapper code below uses the proper
cascade — keep it that way.)

Reference: https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
"""

import torch


class GRUCellAug(torch.nn.Module):
    """A single GRU cell. Numerically equivalent to ``torch.nn.GRUCell`` with
    ``bias=False`` when weights are copied across.
    """

    def __init__(self, input_size, hidden_size, bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # 3x because reset, update, candidate share the same Linear projection
        self.x2h = torch.nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        # See module docstring re: split vs chunk vs slicing.
        gate_x_chunks = gate_x.split(self.hidden_size, dim=1)
        gate_h_chunks = gate_h.split(self.hidden_size, dim=1)
        i_r, i_z, i_n = gate_x_chunks[0], gate_x_chunks[1], gate_x_chunks[2]
        h_r, h_i, h_n = gate_h_chunks[0], gate_h_chunks[1], gate_h_chunks[2]

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_i)
        candidate_h = torch.tanh(i_n + (reset_gate * h_n))

        # Algebraically equivalent to z * h_last + (1 - z) * candidate but
        # avoids the (1 - z) sub which exports cleaner.
        hy = candidate_h + update_gate * (hidden - candidate_h)
        return hy


_SUPPORTED_LAYERS = (1, 2, 3)


class GRUAug(torch.nn.Module):
    """1, 2, or 3 layer stacked GRU with the ESP-DL-friendly export shape.

    Numerically equivalent to ``torch.nn.GRU(num_layers=layers, bias=False)``
    after layer-by-layer weight transplant. Hidden state shape is
    ``(layers, batch, hidden_size)``.

    The 1-3 layer cap is structural — each layer is a named ``Module``
    (``self.l0`` / ``self.l1`` / ``self.l2``) so ONNX traces it as a fixed
    graph rather than a Python loop. If you need >3 layers, add ``self.l3``
    etc. and extend ``forward()``; for the QuaidSIM-v4 use case 1-3 cover
    every trained policy.
    """

    def __init__(self, input_size, hidden_size, bias=False, layers=3):
        super().__init__()
        if layers not in _SUPPORTED_LAYERS:
            raise ValueError(
                f'GRUAug supports {_SUPPORTED_LAYERS} layers, got {layers}. '
                'See the module docstring for how to extend.'
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.layers = layers

        self.l0 = GRUCellAug(input_size, hidden_size, bias)
        if layers >= 2:
            self.l1 = GRUCellAug(hidden_size, hidden_size, bias)
        if layers >= 3:
            self.l2 = GRUCellAug(hidden_size, hidden_size, bias)

    def forward(self, x, hidden):
        # `split(1, dim=0)` yields one (1, B, H) chunk per layer via Split,
        # which ESP-DL supports. See module docstring for the alternatives
        # that don't.
        chunks = hidden.split(1, dim=0)

        # Standard cascade: layer N takes the *current* layer-(N-1) output as
        # its input, and the previous-timestep layer-N hidden state as its
        # carry-in. Matches torch.nn.GRU exactly.
        hidden_0 = self.l0(x, chunks[0].squeeze(0))
        if self.layers == 1:
            return hidden_0, torch.stack((hidden_0,))

        hidden_1 = self.l1(hidden_0, chunks[1].squeeze(0))
        if self.layers == 2:
            return hidden_1, torch.stack((hidden_0, hidden_1))

        # layers == 3
        hidden_2 = self.l2(hidden_1, chunks[2].squeeze(0))
        # `stack` -> Concat (supported). `hidden[i] = ...` -> ScatterND (not).
        return hidden_2, torch.stack((hidden_0, hidden_1, hidden_2))
