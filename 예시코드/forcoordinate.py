import wlkatapython
import serial
import time
import keyboard

serial1 = serial.Serial("COM3", 115200)
mirobot1 = wlkatapython.Wlkata_UART()
mirobot1.init(serial1, -1)
mirobot1.speed(200)
#펜 높이 좌표(오프셋) Z = 112.7

path = [
    (230, 0, 200, 0, 0, 0),
    (160, 0, 200, 0, 0, 0),
    (230, 0, 200, 0, 0, 0),
    (160, 0, 200, 0, 0, 0)
]

mirobot1.zero()

for coord in path:
    x, y, z, rx, ry, rz = coord
    mirobot1.writecoordinate(0, 0, x, y, z, rx, ry, rz)
    time.sleep(1)
    
    coordinates = [mirobot1.getcoordinate(i) for i in range(1, 7)]
    print("Coordinates:", coordinates)
    
