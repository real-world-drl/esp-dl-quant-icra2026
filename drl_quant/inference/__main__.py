"""Command-line entry: ``python -m drl_quant.inference``.

Loads a quantised actor against the QuaidEnv (over MQTT) and runs N evaluation
episodes, printing per-episode and aggregate statistics. For most use cases
all you need is::

    python -m drl_quant.inference \\
        --model models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \\
        --env-config quaid-env/examples/quaid-icra-sim.yaml \\
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
                        help='YAML config for QuaidEnv (e.g. quaid-env/examples/quaid-icra-sim.yaml).')
    parser.add_argument('-q', '--mqtt-queue',
                        help='Override mqtt_queue_no from the YAML (used in MQTT topic '
                             'strings: quaid/obs/r{q}BIN, quaid/act/r{q}, etc.). Matches '
                             'the sim-to-real-cpp player -q flag.')
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
    except ImportError as e:
        # Surface the actual error rather than blanket-blaming quaid_env —
        # a missing transitive dep (gymnasium / pyyaml / paho-mqtt) raises
        # ImportError too and would otherwise be misreported as "package
        # missing".
        sys.stderr.write(f'ERROR: failed to import quaid_env: {e}\n\n')
        missing = getattr(e, 'name', '') or ''
        if missing in ('quaid_env', ''):
            sys.stderr.write(
                'The quaid_env package itself is not importable. Install it '
                'from the sibling directory:\n'
                '   pip install -e quaid-env/\n'
            )
        else:
            sys.stderr.write(
                f'quaid_env is installed but its dependency {missing!r} is not. '
                'Re-install quaid_env so pip pulls its deps:\n'
                '   pip install -e quaid-env/\n'
            )
        sys.stderr.write(
            '\nIf you have multiple Python envs (e.g. conda + venv), confirm '
            'you are running from the one where you installed it:\n'
            '   which python\n'
            '   python -c "import quaid_env; print(quaid_env.__file__)"\n'
        )
        return 2

    from drl_quant.inference.player import Player

    settings = load_settings(args.env_config)
    if args.mqtt_queue is not None:
        # Coerce to str — the YAML loader already does this for the loaded
        # value; we mirror it so a CLI int (e.g. -q 100) goes through the
        # same code path.
        settings.ports.mqtt_queue_no = str(args.mqtt_queue)
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
