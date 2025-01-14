import time
import sys
import threading
import dynamixel_sdk as dxl
import ctypes
import numpy as np

import rby1_sdk
import argparse

# Dynamixel Control Table addresses
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_POSITION = 116
ADDR_GOAL_CURRENT = 102
ADDR_OPERATING_MODE = 11

# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting
BAUDRATE = 2000000

DEVICENAME = "/dev/rby1_gripper"

# Define constants
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
CURRENT_CONTROL_MODE = 0
VELOCITY_CONTROL_MODE = 1
POSITION_CONTROL_MODE = 3
CURRENT_BASED_POSITION_CONTROL_MODE = 5

MIN_INDEX = 0
MAX_INDEX = 1

DEG_PER_PULSE = 360 / 4096  # 0.088

REV_RANGE_MIN = -256
REV_RANGE_MAX = 256

PULSE_RANGE_MIN = REV_RANGE_MIN * 360 / DEG_PER_PULSE
PULSE_RANGE_MAX = REV_RANGE_MAX * 360 / DEG_PER_PULSE

IS_INIT_FINISH = False

PRINT = False


def read_operation_mode(port_handler, packet_handler, dxl_id):
    # data_read, result, error
    operation_mode, dxl_comm_result, dxl_error = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_OPERATING_MODE)
    if dxl_comm_result == dxl.COMM_SUCCESS:
        return operation_mode
    else:
        return None


def read_encoder(port_handler: dxl.PortHandler, packet_handler: dxl.PacketHandler, dxl_id):
    dxl_present_position, dxl_comm_result, dxl_error = packet_handler.read4ByteTxRx(port_handler, dxl_id, ADDR_PRESENT_POSITION)
    dxl_present_position = ctypes.c_int32(dxl_present_position).value
    if PRINT:
        print(f"id: {dxl_id}, Encoder(pulse): {dxl_present_position}, Encoder(deg): {dxl_present_position * DEG_PER_PULSE}, dxl_error: {dxl_error}")
    if dxl_comm_result == dxl.COMM_SUCCESS:
        return dxl_present_position * DEG_PER_PULSE
    else:
        if PRINT:
            print("NONE NONE NONE NONE")
        return None


def send_torque_enable(port_handler, packet_handler, dxl_id, enable):
    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, enable)
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print("%s" % packet_handler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packet_handler.getRxPacketError(dxl_error))
    else:
        if PRINT:
            print(f"ID {dxl_id} has been successfully connected")
    time.sleep(1)


def send_operation_mode(port_handler: dxl.PortHandler, packet_handler: dxl.PacketHandler, dxl_id, mode):
    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_OPERATING_MODE, mode)
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print("%s" % packet_handler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packet_handler.getRxPacketError(dxl_error))
    else:
        if PRINT:
            print(f"(ID: {dxl_id}, Mode: {mode}) has been successfully changed")
    time.sleep(0.5)


def send_current(port_handler: dxl.PortHandler, packet_handler: dxl.PacketHandler, dxl_id, current):
    current_value = int(current / 2.69 * 1000)
    # packet_handler.write2ByteTxOnly(port_handler, dxl_id, ADDR_GOAL_CURRENT, current_value)
    dxl_comm_result, dxl_error = packet_handler.write2ByteTxRx(port_handler, dxl_id, ADDR_GOAL_CURRENT, current_value)
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print("%s" % packet_handler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packet_handler.getRxPacketError(dxl_error))


def send_goal_position(port_handler: dxl.PortHandler, packet_handler: dxl.PacketHandler, dxl_id, goal_position):
    packet_handler.write4ByteTxOnly(port_handler, dxl_id, ADDR_GOAL_POSITION, goal_position)
    time.sleep(0.005)


def control_loop_for_gripper(
    port_handler: dxl.PortHandler, packet_handler: dxl.PacketHandler,
    active_ids, goal_pcts, goal_positions, cur_pcts, cur_positions
):
    q_min_max_vector = [np.zeros(2) for _ in range(2)]
    global IS_INIT_FINISH

    for dxl_id in active_ids:
        q_min_max_vector[dxl_id][MIN_INDEX] = REV_RANGE_MAX * 360
        q_min_max_vector[dxl_id][MAX_INDEX] = REV_RANGE_MIN * 360

    cnt = 0
    while True:
        is_init = True
        for dxl_id in active_ids:
            # Ensure the control mode is in current control mode

            while True:
                operation_mode = read_operation_mode(port_handler, packet_handler, dxl_id)
                if operation_mode is not None:
                    if operation_mode != CURRENT_CONTROL_MODE:
                        send_torque_enable(port_handler, packet_handler, dxl_id, TORQUE_DISABLE)
                        send_operation_mode(port_handler, packet_handler, dxl_id, CURRENT_CONTROL_MODE)
                        send_torque_enable(port_handler, packet_handler, dxl_id, TORQUE_ENABLE)
                    else:
                        break
                time.sleep(0.1)

            # Enable the torque if disabled
            while True:
                torque_enable_val, result, _ = packet_handler.read1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE)
                if torque_enable_val == 0:
                    send_torque_enable(port_handler, packet_handler, dxl_id, TORQUE_ENABLE)
                else:
                    break
                time.sleep(0.1)

            q = read_encoder(port_handler, packet_handler, dxl_id)
            if q is not None:
                if q_min_max_vector[dxl_id][MIN_INDEX] > q:
                    q_min_max_vector[dxl_id][MIN_INDEX] = q
                if q_min_max_vector[dxl_id][MAX_INDEX] < q:
                    q_min_max_vector[dxl_id][MAX_INDEX] = q

            if cnt % 2 == 0:
                send_current(port_handler, packet_handler, dxl_id, 0.5)
            else:
                send_current(port_handler, packet_handler, dxl_id, -0.5)

            # Check if initialization is complete
            if abs(q_min_max_vector[dxl_id][MAX_INDEX] - q_min_max_vector[dxl_id][MIN_INDEX]) < 540 * 0.9:
                is_init = False

            if PRINT:
                print(f" id: {dxl_id}")
                print(f" min val: {q_min_max_vector[dxl_id][MIN_INDEX]}")
                print(f" max val : {q_min_max_vector[dxl_id][MAX_INDEX]}")
                print(f" is_init: {is_init}")

        if is_init:
            for dxl_id in active_ids:
                send_current(port_handler, packet_handler, dxl_id, 0)
            time.sleep(3)
            IS_INIT_FINISH = True
            break

        cnt += 1
        time.sleep(3)

    while True:
        count = 0
        for dxl_id in active_ids:
            operation_mode = read_operation_mode(port_handler, packet_handler, dxl_id)
            if operation_mode is not None:
                if operation_mode != CURRENT_BASED_POSITION_CONTROL_MODE:
                    send_torque_enable(port_handler, packet_handler, dxl_id, TORQUE_DISABLE)
                    send_operation_mode(port_handler, packet_handler, dxl_id, CURRENT_BASED_POSITION_CONTROL_MODE)
                    send_torque_enable(port_handler, packet_handler, dxl_id, TORQUE_ENABLE)
                else:
                    goal_position_id = 0
                    goal_pct_id = max(0., min(1., goal_pcts[dxl_id]))
                    gripper_direction = 0
                    if gripper_direction:
                        goal_position_id = goal_pct_id * q_min_max_vector[dxl_id][MAX_INDEX] + \
                                           (1. - goal_pct_id) * q_min_max_vector[dxl_id][MIN_INDEX]
                    else:
                        goal_position_id = goal_pct_id * q_min_max_vector[dxl_id][MIN_INDEX] + \
                                           (1. - goal_pct_id) * q_min_max_vector[dxl_id][MAX_INDEX]
                    # if count % 100 == 0:
                    #    print(f'{dxl_id=} {goal_pct_id=} {goal_position_id=} ({goal_position_id / DEG_PER_PULSE})')
                    #    q = read_encoder(port_handler, packet_handler, dxl_id)
                    goal_positions[dxl_id] = int(goal_position_id)
                    cur_positions[dxl_id] = read_encoder(port_handler, packet_handler, dxl_id)
                    cur_pcts[dxl_id] = (cur_positions[dxl_id] - q_min_max_vector[dxl_id][MAX_INDEX]) / (
                            q_min_max_vector[dxl_id][MIN_INDEX] - q_min_max_vector[dxl_id][MAX_INDEX])
                    send_goal_position(port_handler, packet_handler, dxl_id, int(goal_position_id / DEG_PER_PULSE))
        count += 1
        time.sleep(0.01)


def pre_process(robot):
    print("Gripper controller attempting to connect to the robot...")

    if not robot.connect():
        print("Error: Unable to establish connection to the robot at")
        sys.exit(1)

    print("Gripper controller successfully connected to the robot")

    if not robot.is_power_on('.*'):
        if not robot.power_on('.*'):
            print("Error")
            return 1
    else:
        print("Power is already ON")
    time.sleep(3)
    if robot.is_power_on('48v'):
        robot.set_tool_flange_output_voltage('right', 12)
        robot.set_tool_flange_output_voltage('left', 12)
        print('Attempting to 12V power on for gripper')
        time.sleep(3)
    return robot


class RainbowGripperController:
    def __init__(self, robot):
        global IS_INIT_FINISH
        self.robot = pre_process(robot)
        # Initialize PortHandler and PacketHandler
        self.port_handler_gripper = dxl.PortHandler(DEVICENAME)
        self.packet_handler_gripper = dxl.PacketHandler(PROTOCOL_VERSION)
        print(f"Protocol Version: {self.packet_handler_gripper.getProtocolVersion()}")
        if not self.port_handler_gripper.openPort():
            print("Failed to open port")
            quit()

        if not self.port_handler_gripper.setBaudRate(BAUDRATE):
            print("Failed to set baud rate")
            quit()

        self.active_ids_gripper = []

        for dxl_id in range(2):
            model_num, dxl_comm_result, err = self.packet_handler_gripper.ping(self.port_handler_gripper, dxl_id)
            if dxl_comm_result == dxl.COMM_SUCCESS:
                print(f"Dynamixel ID {dxl_id} is active")
                self.active_ids_gripper.append(dxl_id)
            else:
                print(f"Dynamixel ID {dxl_id} is not active")

        if len(self.active_ids_gripper) != 2:
            print("Unable to ping all devices for grippers")
            quit()

        for dxl_id in self.active_ids_gripper:
            send_torque_enable(self.port_handler_gripper, self.packet_handler_gripper, dxl_id, TORQUE_DISABLE)
            send_operation_mode(self.port_handler_gripper, self.packet_handler_gripper, dxl_id, CURRENT_CONTROL_MODE)
            send_torque_enable(self.port_handler_gripper, self.packet_handler_gripper, dxl_id, TORQUE_ENABLE)

        self.goal_pcts = np.ones(2)  # Starting value
        self.goal_positions = np.zeros(2)
        self.cur_pcts = np.ones(2)
        self.cur_positions = np.zeros(2)
        self.hand_to_dxl_id = {'right': 0, 'left': 1}
        self.pos_range = 540
        self.control_thread = threading.Thread(target=control_loop_for_gripper, args=(
            self.port_handler_gripper,
            self.packet_handler_gripper,
            self.active_ids_gripper,
            self.goal_pcts,
            self.goal_positions,
            self.cur_pcts,
            self.cur_positions
        ))
        self.control_thread.start()
        while not IS_INIT_FINISH:
            time.sleep(0.005)
        print('Finished initializing gripper controller.')

    def wait_for_gripper(self, hand, max_error_pct=0.05):
        dxl_id = self.hand_to_dxl_id.get(hand)
        assert dxl_id is not None
        for _ in range(100):
            if abs(self.cur_positions[dxl_id] - self.goal_positions[dxl_id]) < max_error_pct * self.pos_range:
                return True
            time.sleep(0.01)
        print('Failed to reach gripper target')
        return False

    def get_gripper_width(self, hand):
        dxl_id = self.hand_to_dxl_id.get(hand)
        return 0.1 * (self.cur_pcts[dxl_id])

    def set_gripper_percent(self, hand, pct, wait=True):
        print('Commanding gripper', hand, pct)
        dxl_id = self.hand_to_dxl_id.get(hand)
        assert dxl_id is not None
        self.goal_pcts[dxl_id] = pct
        if wait:
            success = self.wait_for_gripper(hand)
            return success
        return True

    def set_gripper_open(self, hand, width_in_m, wait=True):
        # Max opening is 100mm = 10cm = 0.1m
        pct = width_in_m / 0.1
        return self.set_gripper_percent(hand, pct, wait=wait)

    def finish(self):
        self.control_thread.join()
        self.port_handler_gripper.closePort()

        self.robot.set_tool_flange_output_voltage("left", 0)
        self.robot.set_tool_flange_output_voltage("right", 0)

