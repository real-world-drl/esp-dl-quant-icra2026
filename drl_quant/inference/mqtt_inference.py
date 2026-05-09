"""Host-side MQTT inference loop (WORK IN PROGRESS).

This module is intended to consume observations streamed over MQTT from a
running Quaid robot, run them through an ONNX actor, and publish actions
back. It currently sets up the MQTT client + SQLite logger; the inference
hook (`parse_observations`) and the action-publish path are still stubs.

Topic layout::

    quaid/obs/r<queue>BIN     # binary observation packets
    quaid/mocap/r<queue>BIN   # binary mocap packets (pose, orientation)
    quaid/set/r<queue>        # text settings updates

Run as::

    python -m drl_quant.inference.mqtt_inference \\
        -e quad -q 100 --description "QuaidSIM-v4 RA-TD3 quantized" \\
        --title "smoke test"
"""

import argparse
import sqlite3
import struct
from datetime import datetime

import paho.mqtt.client as mqtt


class QuaidInference:
    def __init__(self, args):
        self.args = args

        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(args.mqtt_server, 1883, 60)

        t = datetime.now()
        self.logger_name = f'{t.strftime("%Y-%m-%d_%H-%M-%S")}_{args.experiment}_{args.logger_number}'
        self.sqlite = sqlite3.connect(f'data/logger/{self.logger_name}.sqlite', check_same_thread=False)

        self.cursor = self.sqlite.cursor()
        self.cursor.execute('create table readme (description text, title text);')
        self.cursor.execute(
            'INSERT INTO readme (description, title) VALUES (?, ?)',
            (args.description, args.title),
        )
        self.sqlite.commit()

        # TODO: align these with the actual Quaid telemetry packet layouts.
        self.obs_struct = struct.Struct('= B q I ')
        self.mocap_struct = struct.Struct('= B q I f f f')

        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print(f'MQTT connected (rc={rc})')
        client.subscribe(f'quaid/obs/r{self.args.mqtt_queue_no}BIN')
        client.subscribe(f'quaid/mocap/r{self.args.mqtt_queue_no}BIN')
        client.subscribe(f'quaid/set/r{self.args.mqtt_queue_no}')

    def on_message(self, client, userdata, msg):
        if msg.topic.startswith('quaid/obs/r'):
            self.parse_observations(msg)
        elif msg.topic.startswith('quaid/mocap/r'):
            self.parse_mocap(msg.payload.decode())
        elif msg.topic.startswith('quaid/set/r'):
            self.parse_settings(msg)

    def parse_mocap(self, line):
        # TODO: replace ad-hoc parser with the real mocap packet layout
        # and feed pose into the actor's observation builder.
        values = line.split(',')
        # values[1..3] = position, values[4..6] = euler

    def parse_settings(self, msg):
        msg.payload.decode().split(',')

    def parse_observations(self, msg):
        # TODO: unpack observation payload, run through the loaded ONNX model,
        # log to SQLite, publish action back to the robot.
        self.obs_struct.unpack(msg.payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '-e', '--experiment', required=True,
        choices=['fc', 'fcb', 'cc', 'ccb', 'wc', 'wcb', 'wcs',
                 'single_rotor_cc', 'single_rotor_w',
                 'dual_rotor_cc', 'dual_rotor_w',
                 'quad_mounted', 'quad', 'prop_test', 'stepping'],
        help='Experiment kind.',
    )
    parser.add_argument('-q', '--mqtt_queue_no', default=0, help='MQTT queue number.')
    parser.add_argument('-l', '--log_file', help='Pre-recorded log to replay (optional).')
    parser.add_argument('-b', '--buffer_size', default=15000, type=int)
    parser.add_argument('-d', '--description', default='No description provided')
    parser.add_argument('-t', '--title', default='No title provided')
    parser.add_argument('-ms', '--mqtt_server', default='mqtt-server')
    parser.add_argument('-n', '--logger_number', default='0', help='Logger instance number.')
    args = parser.parse_args()

    if args.log_file is not None:
        args.buffer_size = 10_000_000

    QuaidInference(args)


if __name__ == '__main__':
    main()
