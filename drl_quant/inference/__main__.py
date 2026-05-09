"""Command-line entry: ``python -m drl_quant.inference``.

Loads a quantised actor against the QuaidEnv (over MQTT) and runs N evaluation
episodes, printing per-episode and aggregate statistics. For most use cases
all you need is::

    python -m drl_quant.inference \\
        --model models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \\
        --env-config quaid_env/examples/quaid-icra-sim.yaml \\
        --episodes 5

The runner + preprocessor are auto-detected from the model filename; pass
``--runner`` / ``--preprocessor`` to force a particular combination.
"""

from __future__ import annotations

import argparse
import logging
import sys


def get_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m drl_quant.inference',
        description='Evaluate a TD3 / SAC actor against the Quaid env (MQTT).',
    )
    parser.add_argument('-m', '--model', required=True,
                        help='Path to the actor model (.onnx / .pt / .dat).')
    parser.add_argument('-c', '--env-config', required=True,
                        help='YAML config for QuaidEnv (e.g. quaid_env/examples/quaid-icra-sim.yaml).')
    parser.add_argument('-g', '--gru-path',
                        help='Sibling GRU file for recurrent actors that do NOT bake the GRU in. '
                             '(.dat / .pt / .onnx). Ignored for aug_*/with_gru_* ONNXes.')
    parser.add_argument('-e', '--episodes', type=int, default=5)
    parser.add_argument('-s', '--max-steps', type=int, default=500)
    parser.add_argument('--step-delay-ms', type=int, default=0,
                        help='Optional extra sleep between steps (env step_time already governs the rate).')
    parser.add_argument('--rnn-layers', type=int, default=3)
    parser.add_argument('--rnn-hidden-size', type=int, default=64)
    parser.add_argument('--runner', choices=['traced', 'onnx', 'onnx_with_rnn'],
                        help='Override runner auto-detection.')
    parser.add_argument('--preprocessor', choices=['none', 'add_actions', 'gru', 'onnx_gru'],
                        help='Override preprocessor auto-detection.')
    parser.add_argument('--output-dir',
                        help='Directory for SQLite inference-time log + summary.')
    parser.add_argument('--device', default='cpu',
                        help='Torch device for TorchScript paths.')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = get_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    )

    # Imported lazily so ``python -m drl_quant.inference --help`` works
    # even if quaid_env isn't installed.
    try:
        from quaid_env import QuaidEnv, load_settings
    except ImportError:
        sys.stderr.write(
            'ERROR: quaid_env not installed. Install it from the sibling package:\n'
            '   pip install -e quaid_env/\n',
        )
        return 2

    from drl_quant.inference.player import Player

    settings = load_settings(args.env_config)
    env = QuaidEnv(settings)
    env.connect()

    player = Player(
        env,
        args.model,
        gru_path=args.gru_path,
        rnn_layers=args.rnn_layers,
        rnn_hidden_size=args.rnn_hidden_size,
        runner_kind=args.runner,
        preprocessor_kind=args.preprocessor,
        output_dir=args.output_dir,
        test_episodes=args.episodes,
        max_test_steps=args.max_steps,
        test_step_delay_ms=args.step_delay_ms,
        device=args.device,
    )

    try:
        player.play()
        player.stats.print_summary()
    finally:
        player.close()
        env.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
