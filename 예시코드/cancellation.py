import wlkatapython
import serial

serial1 = serial.Serial("COM3", 115200)
mirobot1 = wlkatapython.Wlkata_UART()
mirobot1.init(serial1, -1)

mirobot1.writeangle(1,0,-10,0,0,0,0)
mirobot1.writeangle(1,0,10,0,0,0,0)
#mirobot1.cancellation()

