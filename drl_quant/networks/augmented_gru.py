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


class GRUAug(torch.nn.Module):
    """3-layer stacked GRU with the ESP-DL-friendly export shape."""

    def __init__(self, input_size, hidden_size, bias=False, layers=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.l0 = GRUCellAug(input_size, hidden_size, bias)
        self.l1 = GRUCellAug(hidden_size, hidden_size, bias)
        self.l2 = GRUCellAug(hidden_size, hidden_size, bias)

    def forward(self, x, hidden):
        # `split(1, dim=0)` yields three (1, B, H) chunks via Split, which
        # ESP-DL supports. See module docstring for the alternatives that
        # don't.
        chunks = hidden.split(1, dim=0)
        hidden_0 = self.l0(x, chunks[0].squeeze(0))
        hidden_1 = self.l1(chunks[0].squeeze(0), chunks[1].squeeze(0))
        hidden_2 = self.l2(chunks[1].squeeze(0), chunks[2].squeeze(0))

        # `stack` -> Concat (supported). `hidden[i] = ...` -> ScatterND (not).
        new_hidden = torch.stack((hidden_0, hidden_1, hidden_2))
        return hidden_2, new_hidden
