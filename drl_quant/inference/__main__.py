"""Command-line entry: ``python -m drl_quant.inference``.

Loads a quantised actor against the QuaidEnv (over MQTT) and runs N
evaluation episodes, printing per-episode and aggregate statistics. For
most use cases all you need is::

    python -m drl_quant.inference \\
        --model models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \\
        --env-config quaid-env/examples/quaid-icra-sim.yaml \\
        --episodes 5

The runner + preprocessor are auto-detected from the model filename; pass
``--runner`` / ``--preprocessor`` to force a particular combination.

## Run output

Each invocation creates a timestamped subdirectory matching the C++
``HyperParams::init_snapshot_dir`` convention::

    <output-root>/<env-name>/<policy-name>/<YYYY-MM-DDTHH-MM-SS>/
        Quaid_<timestamp>.sqlite       # observations / actions / rewards / theta_updates
        inference_times.db             # per-step μs + per-episode summary

``--output-root`` defaults to ``data/snapshots`` (matches the C++ default).
``--env-name`` and ``--policy-name`` are auto-detected from the model
filename when not supplied: ``aug_act_net_QuaidSIM-v4_RA-TD3_*.onnx`` ->
env=``QuaidSIM-v4``, policy=``A-TD3``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path


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
    parser.add_argument('--output-root', default='data/snapshots',
                        help='Parent directory for the per-run timestamped folder. '
                             'Default matches the C++ HyperParams::snapshot_dir convention.')
    parser.add_argument('--env-name', default=None,
                        help='Used as the second-level folder name. Defaults to the env '
                             'name parsed from the model filename (e.g. QuaidSIM-v4).')
    parser.add_argument('--policy-name', default=None,
                        help='Used as the third-level folder name. Defaults to the policy '
                             'parsed from the model filename: TD3 / SAC for non-recurrent, '
                             'R-/RA- for external-GRU TorchScript, A- for Aug-GRU ONNX.')
    parser.add_argument('--no-logger', action='store_true',
                        help='Disable per-step SQLite logging (matches C++ env_logger=false). '
                             'Inference timing summary still prints to stdout.')
    parser.add_argument('--device', default='cpu',
                        help='Torch device for TorchScript paths.')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args(argv)


def detect_env_name(model_path: str) -> str:
    """Pull the env identifier out of a model filename.

    ``act_net_QuaidSIM-v4_TD3_+225.dat`` -> ``QuaidSIM-v4``.
    ``aug_act_net_QuaidSIM-v4_RA-TD3_+...onnx`` -> ``QuaidSIM-v4``.
    Falls back to ``unknown`` if the filename doesn't follow the convention.
    """
    name = Path(model_path).name
    m = re.search(r'(?:aug_|with_gru_|with_rnn_)?act_net_([^_]+)_', name)
    return m.group(1) if m else 'unknown'


def detect_policy_name(model_path: str) -> str:
    """Pull the policy tag out of a model filename, matching the C++
    ``-p`` flag convention used in scripts/exports_icra.sh.

    * ``aug_act_net_*_RA-TD3_*`` / ``with_gru_act_net_*_RA-SAC_*`` ->
      ``A-TD3`` / ``A-SAC`` — Aug-GRU is baked in, prepended actions
      handled by the AddActionsPreprocessor; the C++ player calls these
      ``A-`` regardless of whether the source was R- or RA-.
    * ``act_net_*_RA-TD3_*`` -> ``RA-TD3`` (recurrent, external GRU).
    * ``act_net_*_R-TD3_*``  -> ``R-TD3``  (recurrent, no actions).
    * ``act_net_*_TD3_*``    -> ``TD3``    (non-recurrent).
    """
    from drl_quant.inference.player import detect_policy_variant
    info = detect_policy_variant(model_path)
    algo = info['algorithm'] or 'unknown'
    if info['has_gru_inside']:
        return f'A-{algo}'
    if info['is_recurrent']:
        prefix = 'RA-' if info['actions_to_rnn'] else 'R-'
        return f'{prefix}{algo}'
    return algo


def build_run_dir(output_root: str, env_name: str, policy_name: str,
                  timestamp: str | None = None) -> Path:
    """Build ``<root>/<env>/<policy>/<timestamp>/`` and ensure it exists.

    Mirrors HyperParams::init_snapshot_dir from the C++ project.
    """
    if timestamp is None:
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S')
    run_dir = Path(output_root) / env_name / policy_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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

    # Per-run output folder (matches C++ snapshot_dir convention).
    env_name = args.env_name or detect_env_name(args.model)
    policy_name = args.policy_name or detect_policy_name(args.model)
    timestamp = time.strftime('%Y-%m-%dT%H-%M-%S')
    run_dir = build_run_dir(args.output_root, env_name, policy_name, timestamp)
    print(f'Run output: {run_dir}')

    settings = load_settings(args.env_config)
    if args.mqtt_queue is not None:
        # Coerce to str — the YAML loader already does this for the loaded
        # value; we mirror it so a CLI int (e.g. -q 100) goes through the
        # same code path.
        settings.ports.mqtt_queue_no = str(args.mqtt_queue)
    env = QuaidEnv(settings)
    env.connect()

    if not args.no_logger:
        env.setup_logger(run_dir / f'Quaid_{timestamp}.sqlite')

    player = Player(
        env,
        args.model,
        gru_path=args.gru_path,
        rnn_layers=args.rnn_layers,
        rnn_hidden_size=args.rnn_hidden_size,
        runner_kind=args.runner,
        preprocessor_kind=args.preprocessor,
        output_dir=str(run_dir),
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
