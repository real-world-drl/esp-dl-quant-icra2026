# `drl_quant.inference` — host-side runtime (WORK IN PROGRESS)

Intended driver for running a flashed Quaid robot via MQTT: subscribe to the
robot's observation/mocap topics, run the (host-side) ONNX actor, publish
actions, and log everything to SQLite for later analysis.

## Status

Wired up:
- MQTT client + topic subscriptions (`quaid/obs/r<q>BIN`, `quaid/mocap/r<q>BIN`,
  `quaid/set/r<q>`)
- SQLite logger with a `readme` table seeded from `--description` / `--title`
- Packet `struct` placeholders for observation / mocap layouts

Stubs (the part that still needs to be finished):
- `parse_observations` — currently just unpacks the header; needs to decode
  the full observation, run it through a loaded ONNX model, log to SQLite,
  and publish the action back over MQTT.
- `parse_mocap` / `parse_settings` — packet layouts not yet aligned with the
  current Quaid telemetry format.
- ONNX session initialization + the model-path CLI argument.

## Run

```bash
python -m drl_quant.inference.mqtt_inference \
    -e quad -q 100 --description "QuaidSIM-v4 RA-TD3 dynamic-quant" \
    --title "smoke test" -ms <broker-host>
```

The MQTT broker host defaults to `mqtt-server`; override with `-ms`.

## Why is this here?

The on-device inference happens on the ESP32-S3 / ESP32-P4 itself (paper's
main repo). This module is for *host-side* experiments: comparing the
dynamic-quant ONNX (`models/.../onnx-quant/*_qd.onnx`) against the on-device
ESP-DL output, replaying logs, and instrumenting runs without touching MCU
firmware.
