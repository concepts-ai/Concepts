#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pybullet_sim_interfaces.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/29/2024
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

"""Interface with Pybullet simulation. It contains:

- Controller interface for Pybullet simulation.
- Perception interface for Pybullet simulation.
"""

import os
import tempfile
import shutil
import atexit
import queue
from typing import Any, Optional, Callable, Tuple, TYPE_CHECKING

from jacinle.utils.printing import colored
from jacinle.utils.network import get_free_port
from jacinle.comm.service import Service, SocketServer, SocketClient
from concepts.dm.crow.interfaces.perception_interface import CrowPerceptionInterface
from concepts.dm.crow.interfaces.controller_interface import CrowSimulationControllerInterface, CrowPhysicalControllerInterface
from concepts.dm.crow.crow_domain import CrowState
from concepts.dm.crowhat.world.manipulator_interface import RobotControllerExecutionFailed
from concepts.simulator.pybullet.client import BulletClient
from concepts.simulator.pybullet.world import BulletSaver

if TYPE_CHECKING:
    from concepts.simulator.mplib.client import MPLibClient

__all__ = [
    'PyBulletSimulationControllerInterface', 'PyBulletPhysicalControllerInterface', 'PyBulletRemoteService',
    'PyBulletRemotePerceptionInterface', 'PybulletRemoteControllerInterface',
    'make_pybullet_remote_interfaces', 'make_pybullet_simulator_tcp_ports', 'make_pybullet_simulator_ipc_ports',
]


class PyBulletSimulationControllerInterface(CrowSimulationControllerInterface):
    def __init__(self, bullet_client: BulletClient, mplib_client: Optional['MPLibClient'] = None):
        super().__init__()

        self._bullet_client = bullet_client
        self._mplib_client = mplib_client
        self._state_getter = None
        self._saved_states = dict()
        self._saved_states_counter = 0

    @property
    def bullet_client(self):
        return self._bullet_client

    @property
    def mplib_client(self):
        return self._mplib_client

    @property
    def saved_states(self):
        return self._saved_states

    def reset(self):
        self._saved_states.clear()
        self._saved_states_counter = 0

    def save_state(self, **kwargs) -> int:
        self._saved_states_counter += 1

        indent = self._saved_states_counter
        saver = self._bullet_client.world.save_world()
        saver.save()
        self._saved_states[indent] = saver
        return indent

    def restore_state(self, state_identifier: int, **kwargs):
        if state_identifier not in self._saved_states:
            raise ValueError(f"State {state_identifier} not found. Note that the state can only be restored once.")

        saver = self._saved_states[state_identifier]
        if isinstance(saver, BulletSaver):
            saver.restore()
            if self._mplib_client is not None:
                self._mplib_client.sync_object_states(self._bullet_client)
        else:
            saver()
        del self._saved_states[state_identifier]

    def restore_state_keep(self, state_identifier: int, action_counter: Optional[int] = None, **kwargs):
        if state_identifier not in self._saved_states:
            raise ValueError(f"State {state_identifier} not found. Note that the state can only be restored once.")

        saver = self._saved_states[state_identifier]
        if isinstance(saver, BulletSaver):
            saver.restore()
            if self._mplib_client is not None:
                self._mplib_client.sync_object_states(self._bullet_client)
        else:
            saver()

        if action_counter is not None:
            self._action_counter = action_counter

    def register_controller(self, name: str, function: Callable):
        super().register_controller(name, function)

    def register_state_getter(self, state_getter: Callable[['PyBulletSimulationControllerInterface'], CrowState]):
        self._state_getter = state_getter

    def get_crow_state(self) -> CrowState:
        if self._state_getter is not None:
            return self._state_getter(self)
        raise NotImplementedError()


class PyBulletPhysicalControllerInterface(CrowPhysicalControllerInterface):
    def __init__(self, bullet_client: BulletClient, dry_run: bool = False):
        super().__init__()
        self._bullet_client = bullet_client
        self._dry_run = dry_run

    @property
    def bullet_client(self):
        return self._bullet_client

    def serve(
        self, *, tcp_ports: Optional[Tuple[int, int]] = None, ipc_ports: Optional[Tuple[str, str]] = None,
        redirect_ios: bool = False,
        redirect_stdout: Optional[str] = '/dev/null', redirect_stderr: Optional[str] = '/dev/null'
    ) -> None:
        service = PyBulletRemoteService(self, dry_run=self._dry_run)
        server = SocketServer(service, 'pybullet-physical-controller-interface', tcp_port=tcp_ports, ipc_port=ipc_ports)
        with server.activate():
            if redirect_ios:
                return service.mainloop(redirect_stdout=redirect_stdout, redirect_stderr=redirect_stderr)
            else:
                return service.mainloop()


class PyBulletRemoteService(Service):
    def __init__(self, controller: PyBulletPhysicalControllerInterface, dry_run: bool = False):
        super().__init__(spec={'controllers': list(controller.controllers.keys())})
        self._controller = controller
        self._dry_run = dry_run
        self.queue = queue.Queue()

    def call(self, action_name, *args, **kwargs):
        # Create a Future to store the result of the action.
        q = queue.Queue(maxsize=1)
        self.queue.put((action_name, args, kwargs, q))
        return q.get()

    def _redirect_output(self, redirect_stdout: Optional[str], redirect_stderr: Optional[str]):
        if redirect_stdout is not None:
            # If file not exists, create it.
            if not os.path.exists(redirect_stdout):
                with open(redirect_stdout, 'w') as f:
                    pass
            os.dup2(os.open(redirect_stdout, os.O_WRONLY), 1)

        if redirect_stderr is not None:
            if not os.path.exists(redirect_stderr):
                with open(redirect_stderr, 'w') as f:
                    pass
            os.dup2(os.open(redirect_stderr, os.O_WRONLY), 2)

    def mainloop(self, *, redirect_stdout: str = '/dev/null', redirect_stderr: str = '/dev/null'):
        # self._redirect_output(redirect_stdout, redirect_stderr)
        while True:
            while True:
                try:
                    action_name, args, kwargs, result_q = self.queue.get(block=False)
                    break
                except queue.Empty:
                    self._controller.bullet_client.update_viewer()
                except KeyboardInterrupt:
                    return
            try:
                if action_name == '__get_scene__':
                    result_q.put({'status': 'done', 'scene': self._controller.bullet_client.world.save_world()})
                else:
                    if self._dry_run:
                        print('Dry run:', action_name, args, kwargs)
                        result_q.put({'status': 'warning', 'message': f'Controller {action_name} is in dry run mode.'})
                    else:
                        self._controller.controllers[action_name](*args, **kwargs)
                        result_q.put({'status': 'done'})
            except RobotControllerExecutionFailed as e:
                result_q.put({'status': 'failed', 'message': e.args})
            except KeyError:
                result_q.put({'status': 'error', 'message': f'Controller {action_name} not found.'})
            except KeyboardInterrupt:
                return


class PyBulletRemotePerceptionInterface(CrowPerceptionInterface):
    def __init__(self, client):
        super().__init__()
        self._client = client
        self._bullet_client = None
        self._state_getter = None

    def get_scene(self):
        rv = self._client.call('__get_scene__')
        if rv['status'] == 'done':
            return rv['scene']
        else:
            raise RuntimeError(f"Failed to get scene: {rv}")

    def register_bullet_client(self, bullet_client: BulletClient):
        self._bullet_client = bullet_client

    def register_state_getter(self, state_getter: Callable[['PyBulletRemotePerceptionInterface'], CrowState]):
        self._state_getter = state_getter

    def step(self, action: Optional[Any] = None) -> None:
        pass

    def update_simulator(self) -> None:
        if self._bullet_client is None:
            raise ValueError("Bullet client not registered.")
        saver = self.get_scene()
        saver.reset_client_id(self._bullet_client.client_id, self._bullet_client.world)
        saver.restore()

    def get_crow_state(self) -> CrowState:
        if self._state_getter is not None:
            return self._state_getter(self)
        raise NotImplementedError()


class PybulletRemoteControllerInterface(CrowPhysicalControllerInterface):
    def __init__(self, client: SocketClient):
        super().__init__()
        self._client = client
        self._spec = self._client.get_spec()
        for name in self._spec['controllers']:
            self.register_controller(name, self._make_controller(name))

    def _make_controller(self, name):
        def controller(*args, **kwargs):
            rv = self._client.call(name, *args, **kwargs)
            print(rv)
            if rv['status'] == 'done':
                return
            elif rv['status'] == 'failed':
                raise RobotControllerExecutionFailed(f"Controller {name} failed: {rv['message']}")
            elif rv['status'] == 'warning':
                print(colored(f"Controller {name} warning: {rv['message']}", 'yellow'))
            elif rv['status'] == 'error':
                raise RobotControllerExecutionFailed(f"Controller {name} failed with system error {rv['message']}")
            else:
                raise RobotControllerExecutionFailed(f"Controller {name} failed with unknown status: {rv}")
        return controller


def make_pybullet_remote_interfaces(*, ipc_ports: Optional[Tuple[str, str]] = None, tcp_ports: Optional[Tuple[int, int]] = None, host: str = '127.0.0.1') -> Tuple[PyBulletRemotePerceptionInterface, PybulletRemoteControllerInterface]:
    if ipc_ports is not None:
        client = SocketClient('pybullet-physical-controller-interface', [f'ipc://{ipc_ports[0]}', f'ipc://{ipc_ports[1]}'], echo=False)
    elif tcp_ports is not None:
        client = SocketClient('pybullet-physical-controller-interface', [f'tcp://{host}:{tcp_ports[0]}', f'tcp://{host}:{tcp_ports[1]}'], echo=False)
    else:
        raise ValueError("Either ipc_ports or tcp_ports should be provided.")

    client.initialize()

    def _atexit():
        print(colored(f"Finalizing the client. {client.name}", 'yellow'))
        client.finalize()

    atexit.register(_atexit)
    return PyBulletRemotePerceptionInterface(client), PybulletRemoteControllerInterface(client)


def make_pybullet_simulator_ipc_ports() -> Tuple[str, str]:
    temp_dir = tempfile.mkdtemp()
    ipc1 = temp_dir + '/ipc1'
    ipc2 = temp_dir + '/ipc2'

    def _atexit():
        print(colored(f"Removing the temporary directory: {temp_dir}", 'yellow'))
        shutil.rmtree(temp_dir)

    atexit.register(_atexit)
    return ipc1, ipc2


def make_pybullet_simulator_tcp_ports() -> Tuple[int, int]:
    return get_free_port(), get_free_port()
