import time
import serial
import wlkatapython

from .config import *


class RobotController:
    def __init__(self):
        self.serial = None
        self.robot = None
        self.connected = False

    def custom_tool_offset(self, x, y, z):
        msg = (
            '$46=' + str(x) + '\n'
            '$47=' + str(y) + '\n'
            '$48=' + str(z)
        )

        self.robot.sendMsg(msg)

    def connect(self):
        if self.connected:
            return

        self.serial = serial.Serial(SERIAL_PORT, BAUD_RATE)
        time.sleep(2)

        self.robot = wlkatapython.Wlkata_UART()
        self.robot.init(self.serial, ROBOT_ADDRESS)
        self.robot.speed(ROBOT_SPEED)
        self.custom_tool_offset(35, 0, 120)
        self.connected = True
        

    def homing(self):
        self.robot.homing()

    def zero(self):
        self.robot.zero()

    def move(self, x, y, z, linear=True):
        x = max(ROBOT_X_MIN, min(ROBOT_X_MAX, float(x)))
        y = max(ROBOT_Y_MIN, min(ROBOT_Y_MAX, float(y)))

        motion = 1 if linear else 0
        position = 0

        self.robot.writecoordinate(
            motion,
            position,
            round(x, 2),
            round(y, 2),
            round(z, 2),
            ROBOT_RX,
            ROBOT_RY,
            ROBOT_RZ,
        )
        return x, y, float(z)

    def get_xyz(self):
        return (
            float(self.robot.getcoordinate(1)),
            float(self.robot.getcoordinate(2)),
            float(self.robot.getcoordinate(3)),
        )

    def stop(self):
        if self.robot is not None:
            self.robot.cancellation()

    def close(self):
        if self.serial is not None:
            try:
                self.serial.close()
            finally:
                self.serial = None
                self.robot = None
                self.connected = False
