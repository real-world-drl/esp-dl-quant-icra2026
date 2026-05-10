"""Filename-based algorithm detection for the ONNX exporters.

The pipeline (export -> quantize -> infer) keys off load-bearing tokens in
the model filename. Exporters need to know whether to instantiate a
``TD3Actor`` or a ``DiagGaussianActor``; the cheapest place to keep that
metadata is the filename itself, so the trained ``.dat`` checkpoints from
the upstream training repo all carry an algorithm tag.

If a checkpoint comes from elsewhere and lacks the tag, we raise a clear
``ValueError`` rather than silently falling through to the SAC head — the
two heads have different output dimensions, so a wrong choice fails late
inside ``load_state_dict`` with an opaque shape-mismatch error.
"""

from __future__ import annotations

from pathlib import Path


_VALID_NAME_EXAMPLES = """\
  act_net_QuaidSIM-v4_TD3_+225.827_750000.dat        (non-recurrent TD3)
  act_net_QuaidSIM-v4_SAC_+238.823_775000.dat        (non-recurrent SAC)
  act_net_QuaidSIM-v4_R-TD3_+364.117_475000.dat      (recurrent TD3)
  act_net_QuaidSIM-v4_RA-SAC_+299.788_850000.dat     (recurrent + actions, SAC)
"""


def detect_algorithm(model_path: str) -> str:
    """Return ``'TD3'`` or ``'SAC'`` based on the filename. Raise
    ``ValueError`` with examples if the convention is not followed."""
    name = Path(model_path).name
    if 'TD3' in name:
        return 'TD3'
    if 'SAC' in name:
        return 'SAC'
    raise ValueError(
        f"Could not detect algorithm from filename: {name!r}.\n\n"
        "The filename must contain either 'TD3' or 'SAC' so the exporter "
        "knows which actor head to instantiate (TD3 has a Tanh head with "
        "action_dim outputs; SAC has a Gaussian head with action_dim*2 "
        "outputs — they are not interchangeable).\n\n"
        "Examples of valid names:\n"
        f"{_VALID_NAME_EXAMPLES}\n"
        "Rename the file to include the algorithm tag and re-run, e.g.:\n"
        f"  mv {name} {Path(name).stem}_TD3{Path(name).suffix}"
    )
