#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualizer.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/20/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Robot state visualizer written using Dash and ZMQ for inter-process communication."""

import pickle
import threading
import time
import zmq

from concepts.hw_interface.robot_state_visualizer.visualizer import RobotStateVisualizer


class RobotStateVisualizerMultiproc(RobotStateVisualizer):
    INIT_PORT = 5557
    SUB_PORT = 5556

    def listener_thread(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f'tcp://localhost:{self.SUB_PORT}')
        socket.setsockopt_string(zmq.SUBSCRIBE, '')

        while True:
            try:
                message = socket.recv()
                message = pickle.loads(message)
                if message['type'] == 'update':
                    payload = message['data']
                    with self.mutex:
                        for key, data in payload.items():
                            timestamp, value = data
                            if isinstance(key, tuple):
                                tab, name = key
                                self.update_queue_with_mutex(name, timestamp, value, tab=tab)
                            else:
                                self.update_queue_with_mutex(key, timestamp, value)
                else:
                    print(f'Unknown message type: {message["type"]}')
            except Exception as e:
                print(f'Error: {e}')
                import traceback
                traceback.print_exc()

    def start(self):
        # Create a REP socket to receive initialization message.
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        print(f'Binding to port {self.INIT_PORT}')
        socket.bind(f'tcp://*:{self.INIT_PORT}')
        print(f'Visualizer server started at port {self.INIT_PORT}.')
        message = socket.recv()
        message = pickle.loads(message)

        self.reset(queues=message['data'])
        self.initialized = True
        socket.send(b'OK')
        socket.close()

        print('Visualizer initialized.')

        threading.Thread(target=self.listener_thread, daemon=True).start()
        super().start()

        while True:
            time.sleep(1)


class RobotStateVisualizerPublisher(object):
    PUB_PORT = 5556

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f'tcp://*:{self.PUB_PORT}')
        self.reset_message_sent = False

    def publish(self, message):
        if not self.reset_message_sent:
            raise ValueError('Reset message not sent before update message.')
        message = {'type': 'update', 'data': message}
        self.socket.send(pickle.dumps(message))

    def reset(self, queues):
        if self.reset_message_sent:
            raise ValueError('Reset message already sent.')
        message = {'type': 'reset', 'data': queues}
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f'tcp://localhost:{RobotStateVisualizerMultiproc.INIT_PORT}')
        socket.send(pickle.dumps(message))
        socket.recv()
        socket.close()
        self.reset_message_sent = True

