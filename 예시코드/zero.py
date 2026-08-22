import wlkatapython
import serial
import time


serial1 = serial.Serial("COM3", 115200)
mirobot1 = wlkatapython.Wlkata_UART()
mirobot1.init(serial1, -1)

mirobot1.sendMsg("M20 G91 X10")
time.sleep(2)
mirobot1.zero()
serial1.close()
