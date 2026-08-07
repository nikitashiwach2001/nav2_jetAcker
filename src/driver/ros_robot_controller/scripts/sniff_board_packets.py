#!/usr/bin/env python3
"""Diagnostic: what packet types does the ros_robot_controller board actually SEND?

Why: the JetAcker has wheel encoders wired to this board, but nothing in the ROS stack
ever sees them. Tracing the driver shows why -- ros_robot_controller_sdk.py registers
report parsers for SYS/KEY/IMU/GAMEPAD/BUS_SERVO/SBUS/PWM_SERVO only. PACKET_FUNC_MOTOR
(=3) exists in the protocol enum but has NO parser, and the SDK has set_motor_speed()
with no matching read. So encoder data is either (a) being sent and silently dropped, or
(b) never sent at all -- and which one it is decides whether this is a small SDK patch or
a firmware-level dead end.

This script bypasses the SDK: it opens the board's serial port directly, decodes the raw
framing (0xAA 0x55 <func> <len> <data...> <crc8>) and just COUNTS what arrives, per
function id. If func 3 (MOTOR) shows up, the encoders are reachable from software.

IMPORTANT: the board's serial port can only be held by one process. Stop the ROS stack
first (the ros_robot_controller node owns /dev/rrc while it runs), e.g.:
    sudo systemctl stop start_app_node.service
    pkill -f ros_robot_controller

Usage:
    python3 sniff_board_packets.py [--seconds 15] [--device /dev/rrc]

Drive the wheels while it runs (push the robot, or use the joystick) -- encoder reports
are usually only emitted when the motors actually turn.
"""
import argparse
import collections
import sys
import time

import serial

FUNC_NAMES = {
    0: 'SYS', 1: 'LED', 2: 'BUZZER', 3: 'MOTOR  <-- encoders would be here',
    4: 'PWM_SERVO', 5: 'BUS_SERVO', 6: 'KEY', 7: 'IMU', 8: 'GAMEPAD',
    9: 'SBUS', 10: 'OLED',
}
PACKET_FUNC_NONE = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='/dev/rrc')
    ap.add_argument('--baud', type=int, default=1000000)
    ap.add_argument('--seconds', type=float, default=15.0)
    args = ap.parse_args()

    try:
        port = serial.Serial(None, args.baud, timeout=0.1)
        port.rts = False
        port.dtr = False
        port.setPort(args.device)
        port.open()
    except Exception as e:
        print(f'cannot open {args.device}: {e}')
        print('Is the ROS stack still running and holding the port? Stop it first.')
        return 1

    print(f'listening on {args.device} @ {args.baud} for {args.seconds:.0f} s ...')
    print('>>> DRIVE THE WHEELS NOW (joystick, or push the robot) <<<\n')

    counts = collections.Counter()
    samples = {}
    # framing state machine, same as ros_robot_controller_sdk.recv_task
    STATE_S1, STATE_S2, STATE_FUNC, STATE_LEN, STATE_DATA, STATE_CRC = range(6)
    state = STATE_S1
    func = length = 0
    data = bytearray()

    end = time.time() + args.seconds
    while time.time() < end:
        chunk = port.read(256)
        for b in chunk:
            if state == STATE_S1:
                if b == 0xAA:
                    state = STATE_S2
            elif state == STATE_S2:
                state = STATE_FUNC if b == 0x55 else STATE_S1
            elif state == STATE_FUNC:
                if b < PACKET_FUNC_NONE:
                    func = b
                    state = STATE_LEN
                else:
                    state = STATE_S1
            elif state == STATE_LEN:
                length = b
                data = bytearray()
                state = STATE_CRC if length == 0 else STATE_DATA
            elif state == STATE_DATA:
                data.append(b)
                if len(data) >= length:
                    state = STATE_CRC
            elif state == STATE_CRC:
                counts[func] += 1
                if func not in samples:
                    samples[func] = bytes(data)
                state = STATE_S1
    port.close()

    print('=== packets received from the board ===')
    if not counts:
        print('NOTHING received. Either the port is wrong, the board is off, or another '
              'process is still holding it.')
        return 1
    for f, n in sorted(counts.items()):
        name = FUNC_NAMES.get(f, f'unknown({f})')
        smp = samples[f]
        print(f'  func {f:2d}  {name:<32} {n:5d} packets   '
              f'{len(smp)} data bytes, first: {smp.hex()[:32]}')

    print()
    if 3 in counts:
        print('RESULT: the board DOES report MOTOR (func 3) packets.')
        print('        -> encoders are reachable from software; the SDK just drops them.')
        print('        Next step: add a report parser + get_motor_speed() to the SDK,')
        print('        publish it as a topic, and fuse it in the EKF.')
    else:
        print('RESULT: NO MOTOR (func 3) packets arrived.')
        print('        -> the board never reports encoder/motor feedback over this')
        print('        protocol. Encoders are then used only inside the board for its own')
        print('        closed-loop speed control, and no SDK change can expose them --')
        print('        it would need different board firmware.')
        print('        (Re-run while the wheels are definitely turning before concluding.)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
