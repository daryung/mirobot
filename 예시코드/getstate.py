import wlkatapython
import serial
import time

serial1 = serial.Serial("COM3", 115200)
mirobot1 = wlkatapython.Wlkata_UART()
mirobot1.init(serial1, -1)


print("State:", mirobot1.getState())


angles = [mirobot1.getAngle(i) for i in range(1, 7)]
print("Angles:", angles)

coordinates = [mirobot1.getcoordinate(i) for i in range(1, 7)]
print("Coordinates:", coordinates)

print("Mode:", mirobot1.getmooe())

time.sleep(2)
serial1.close()

