# `drl_quant.inference`

Evaluate a trained TD3 / SAC actor against the Quaid env (MQTT) and collect
inference-time + reward statistics. Python port of
``sim-to-real-cpp/src/inference/Player.cpp`` minus the on-device
``MCUInference`` path (skipped intentionally — that runs on the ESP32).

## What's here

```
drl_quant/inference/
├── runners.py          backends: TracedRunner, OnnxRunner, OnnxWithRnnRunner
├── preprocessors.py    NoPreprocessor, AddActions, Gru, OnnxGru
├── stats.py            InferenceStats — per-step μs + per-episode reward + sqlite
├── player.py           orchestrator + filename-based auto-dispatch
└── __main__.py         CLI: python -m drl_quant.inference
```

## Supported model formats

| File pattern                         | Backend              | Notes                                                                 |
|--------------------------------------|----------------------|-----------------------------------------------------------------------|
| ``*.pt`` / ``*.dat``                 | ``TracedRunner``     | TorchScript (the C++ training repo writes ``.dat``; same format).      |
| ``act_net_*.onnx``                   | ``OnnxRunner``        | Single-input ONNX. For non-recurrent TD3 / SAC.                       |
| ``aug_act_net_*.onnx``               | ``OnnxWithRnnRunner`` | Aug-GRU baked in. Two inputs (obs + h_t), two outputs (action + h_t). |
| ``with_gru_act_net_*.onnx``          | ``OnnxWithRnnRunner`` | Native ``nn.GRU`` baked in. Same I/O shape as Aug.                    |
| ``*_qd.onnx`` (dynamic-quant)        | same as the source    | onnxruntime int8; otherwise identical.                                |

## Auto-dispatch — what fires when

The Player inspects the model filename and picks a (runner, preprocessor)
pair. The C++ Player uses the same heuristics; we mirror them so you can
swap runtimes 1:1.

```
                                       ┌─ Aug-GRU / native GRU baked into ONNX ────┐
              ┌─ ONNX ─────────────────│                                            │
              │                        └─ no GRU inside (act_net_*.onnx) ──────────┤
─ filename ───┤                                                                    │
              └─ TorchScript (.pt / .dat) ─ actor head only; recurrence external ──┘
```

```
            ┌── R-  (recurrent, no prev action) ────────────────────────────┐
            │                                                                │
─ tag in ───┼── RA- (recurrent + previous action) ─────────────────────────┐ │
  filename  │                                                              │ │
            └── plain TD3 / SAC (non-recurrent) ─────────────────────────┐ │ │
                                                                         ▼ ▼ ▼
```

Results in:

| Filename example                                        | Runner               | Preprocessor              |
|---------------------------------------------------------|----------------------|---------------------------|
| ``act_net_*_TD3_*.onnx``                                | OnnxRunner           | NoPreprocessor            |
| ``act_net_*_SAC_*.dat``                                 | TracedRunner         | NoPreprocessor            |
| ``aug_act_net_*_R-TD3_*.onnx``                          | OnnxWithRnnRunner    | NoPreprocessor            |
| ``aug_act_net_*_RA-SAC_*.onnx``                         | OnnxWithRnnRunner    | AddActionsPreprocessor    |
| ``with_gru_act_net_*_RA-TD3_*.onnx``                    | OnnxWithRnnRunner    | AddActionsPreprocessor    |
| ``act_net_*_R-TD3_*.dat`` + ``rnn_*.dat``                | TracedRunner         | GruPreprocessor           |
| ``act_net_*_RA-SAC_*.dat`` + ``rnn_*.dat``               | TracedRunner         | GruPreprocessor (actions) |
| ``act_net_*_R-TD3_*.onnx`` + ``rnn_*.onnx``              | OnnxRunner           | OnnxGruPreprocessor       |

Override either choice via ``--runner`` / ``--preprocessor``.

## CLI

```bash
# Sim eval — uses queue 100 from quaid-icra-sim.yaml
python -m drl_quant.inference \
    --model models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx \
    --env-config quaid-env/examples/quaid-icra-sim.yaml \
    --episodes 5 \
    --output-dir runs/eval-aug-ra-td3
```

### MQTT queue selection (`-q`)

Real robot and simulator must be on **different MQTT queues** so an eval
run can't accidentally drive the live robot. The bundled YAMLs follow the
project convention:

| Config                 | Default queue | Use                              |
|------------------------|---------------|----------------------------------|
| `quaid-icra-real.yaml` | `99`          | The real robot                    |
| `quaid-icra-sim.yaml`  | `100`         | A simulator instance              |
| (training)             | distinct band | Multiple parallel jobs use disjoint ranges to avoid cross-talk |

Override per-run with `-q` (matches the C++ player) instead of editing the
YAML:

```bash
# Run against a second simulator instance on queue 101
python -m drl_quant.inference \
    --model ... --env-config quaid-env/examples/quaid-icra-sim.yaml -q 101

# Eval against the real robot
python -m drl_quant.inference \
    --model ... --env-config quaid-env/examples/quaid-icra-real.yaml -q 99
```

Recurrent TorchScript with an external GRU:

```bash
python -m drl_quant.inference \
    --model models/QuaidSIM-v4/cpp/act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.dat \
    --gru-path models/rnn/rnn_Quaid_RA-64.dat \
    --env-config quaid-env/examples/quaid-icra-sim.yaml
```

## Programmatic use

```python
from quaid_env import QuaidEnv, load_settings
from drl_quant.inference import Player

env = QuaidEnv(load_settings('quaid-env/examples/quaid-icra-sim.yaml'))
env.connect()

player = Player(
    env,
    'models/QuaidSIM-v4/onnx/aug_act_net_QuaidSIM-v4_RA-TD3_+439.031_450000.onnx',
    test_episodes=5,
    output_dir='runs/eval-1',
)
stats = player.play()
stats.print_summary()
player.close()
env.close()

print(stats.summary()['mean_inference_us'])
```

## Statistics

Per episode:
- reward, step count, wall seconds, FPS
- mean / stdev inference time (microseconds)

Aggregate:
- mean ± stdev reward, mean steps
- inference: mean / stdev / min / max / p50 / p95 / p99 (μs)

If ``--output-dir`` is set, the run also writes
``<output-dir>/inference_times.db`` with two SQLite tables matching the C++
``Trainer::log_inference_times`` schema:

```sql
CREATE TABLE inference_times (id PK, episode_no, step, inference_time_us);
CREATE TABLE episodes        (id PK, episode_no, reward, steps, wall_seconds);
```

## What's intentionally skipped

- ``MCUInference`` — that path runs on the ESP32 itself (the paper's main
  repo). Host-side Python doesn't need it.
- ``FrameStackingPreprocessor`` — the QuaidSIM-v4 policies don't use it; can
  be added later by mirroring `FrameStackingPreprocessor.cpp`.
- The C++ Player's ``params->rnn_a_model`` fallback for missing-snapshot
  GRUs — pass ``--gru-path`` explicitly instead.
