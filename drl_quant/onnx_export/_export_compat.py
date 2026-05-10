"""Cross-version compatibility for ``torch.onnx.export``.

torch >= 2.6 has been migrating ``torch.onnx.export`` toward a new
``dynamo``-based exporter and away from the TorchScript-based legacy path.
The new path:

* fails with ``AttributeError: 'LeafSpec' object has no attribute 'type'``
  on some module shapes (notably ours);
* refuses any ``opset_version`` below 18.

Our exporters target the legacy path on purpose — they explicitly trace
``nn.Module`` graphs that the legacy exporter handles cleanly, and we use
opsets 15 / 17 because that's what ESP-DL and PPQ track. Passing
``dynamo=False`` keeps us on the legacy path on torch >= 2.5; on older
torches the parameter doesn't exist so we omit it.
"""

from __future__ import annotations

import inspect

import torch


_HAS_DYNAMO_PARAM = 'dynamo' in inspect.signature(torch.onnx.export).parameters

# Spread these into every torch.onnx.export(...) call.
LEGACY_EXPORT_KWARGS: dict = {'dynamo': False} if _HAS_DYNAMO_PARAM else {}
