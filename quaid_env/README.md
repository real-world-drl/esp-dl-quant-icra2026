# quaid-env

[Gymnasium](https://gymnasium.farama.org/)-style environment for the Quaid
quadruped robot, communicating over MQTT. Python port of the C++
`QuaidEnv` from the sister `sim-to-real-cpp` project — same observation
layout, same reward, same control wire format, so trained policies behave
identically.

This package is intentionally **independent of any specific actor / policy
framework** (it doesn't pull torch or onnxruntime). You get observations,
you publish actions, that's the whole job. The model loading and inference
loop live in the consumer (e.g. the sibling `drl_quant.inference` module).

It currently lives inside the `esp-dl-quant-icra2026` companion repo for
convenience while it stabilises; it is built as a self-contained
pip-installable package so it can be moved to its own repo with a single
`git mv quaid_env <new-repo>` and no import-rewrite churn.

## Install

```bash
pip install -e .
# or, with tests:
pip install -e .[test]
```

## Quick start

```python
from quaid_env import QuaidEnv, load_settings

settings = load_settings('examples/quaid-icra-sim.yaml')
env = QuaidEnv(settings)
env.connect()                        # opens the MQTT session

obs, info = env.reset()
for step in range(settings.robot.max_steps):
    action = my_policy(obs)          # 8-dim, in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

`obs` is a `numpy.ndarray` of shape `(17,)` and dtype `float32`. `action` is
also 8-dim, clipped to [-1, 1] before publishing.

## Architecture

```
quaid_env/
├── packets.py          struct codecs for the binary obs (0x0A) + mocap (0x0E) packets
├── quaid_data.py       SharedQuaidData — thread-safe merged sensor + mocap state
├── config.py           YAML loader (handles the OpenCV %YAML:1.0 prefix)
├── observations.py     17-dim observation vector builder + normalization
├── reward.py           reward terms + breakdown
├── theta.py            random-yaw frame rotation ("theta adjustment")
├── mqtt_controller.py  paho-mqtt wrapper: subscribe / decode / publish
└── env.py              QuaidEnv(gymnasium.Env) — orchestrator
```

### Observation layout — 17 floats, fixed slot order

```
0  time_delta             OR current_front_left      (if current_per_leg)
1  distance               OR current_front_right     (if current_per_leg)
2  voltage                OR current_back_left       (if current_per_leg)
3  current_total
4  yaw
5  yaw_delta              OR acc_z                   (if acc_z)
6  pitch
7  roll
8  observation_age        OR current_back_right      (if current_per_leg)
9..16  servo positions: BL knee, BL thigh, BR knee, BR thigh,
                        FR knee, FR thigh, FL knee, FL thigh
```

The aliasing (slots 0/1/2/8 reused for per-leg currents, slot 5 reused for
`acc_z`) is intentional and matches the C++ env. The QuaidSIM-v4 config
files in `examples/` enable `current_per_leg` and `acc_z` so the deployed
layout puts the four leg currents at 0/1/2/8 and `acc_z` at 5.

If your config enables `time_delta` / `distance` / `voltage` /
`observation_age` while *also* enabling `current_per_leg`, the per-leg
currents win — they are written last and overwrite. This is faithful to
the C++ behaviour; if you want both, change the slot allocation policy in
`observations.py`.

### Action layout — 8 floats, in [-1, 1]

The 8 policy outputs map to the 4 legs as `(knee, thigh)` pairs in this
order: BL, BR, FR, FL. The MQTT controller expands them to 16 floats on the
wire as `[a0,a1,0,0, a2,a3,0,0, a4,a5,0,0, a6,a7,0,0]` so the on-MCU parser
sees the same format the C++ env uses.

The wire payload is text, published to `quaid/act/r{queue}`:

```
a<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,<f>,0\n
```

The trailing `,0\n` is intentional — it matches a quirk of the C++
`std::ostream_iterator<float>` formatter which always emits a trailing
comma. The MCU's parser depends on that final `0`.

### MQTT topics (parameterised by `mqtt_queue_no`)

| Direction  | Topic                  | Payload                     |
|------------|------------------------|------------------------------|
| subscribe  | `quaid/obs/r{q}BIN`    | binary, 83 bytes, 0x0A header |
| subscribe  | `quaid/mocap/r{q}BIN`  | binary, 37 bytes, 0x0E header |
| subscribe  | `quaid/ctrl/r{q}`      | text — `P0`/`P1` pause, `R` reload, `O<deg>` rotate |
| publish    | `quaid/act/r{q}`       | text — actions + single-char commands (`s`/`r`/`e`/`x`/`y`/`z`/`u`/`i`/`f`/`o`/`n`/`c`) |
| publish    | `quaid/set/r{q}`       | text — used together with ctrl for "message" broadcasts |

## Configuration

The YAML format matches the C++ config (`%YAML:1.0` directive, then four
sections):

- `ports:` — MQTT broker URI, queue number, serial port (unused here)
- `robot:` — step timing, reset strategy, theta-adjustment params, mocap
  bounds, per-queue offsets
- `observations:` — which fields are enabled, normalization mode
- `reward:` — every term's weight, target distance per step, etc.

Sample configs are in `examples/`. `load_settings(path)` strips the
OpenCV `%YAML:1.0` line and returns a populated `Settings` dataclass.

## Run the tests

The unit tests do not require an MQTT broker:

```bash
pip install -e .[test]
pytest
```

## Limitations / known gaps

- The MQTT bring-up is not yet exercised end-to-end against a live robot or
  mosquitto broker — that is the next milestone.
- Sim-only "no MQTT" mode (the C++ `sim: 1` branch) is not implemented; if
  you want sim training, run the simulator and have it publish to MQTT.
- SQLite per-episode logging is not ported. Add it later if needed.
- The `ctrl` topic only handles `P<int>` (pause). `R` (reload) and `O<deg>`
  (mocap rotation offset) are robot-side ops that need no client state.
