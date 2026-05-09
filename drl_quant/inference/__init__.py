"""Inference / evaluation runner for trained TD3 / SAC actors.

Public API::

    from drl_quant.inference import Player, detect_policy_variant
    from drl_quant.inference.runners import (
        TracedRunner, OnnxRunner, OnnxWithRnnRunner,
    )
    from drl_quant.inference.preprocessors import (
        NoPreprocessor, AddActionsPreprocessor,
        GruPreprocessor, OnnxGruPreprocessor,
    )

Or just run from the CLI::

    python -m drl_quant.inference --help
"""

from drl_quant.inference.player import Player, detect_policy_variant

__all__ = ['Player', 'detect_policy_variant']
