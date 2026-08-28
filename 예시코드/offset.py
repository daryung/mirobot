import wlkatapython
import serial
import time

serial1 = serial.Serial("COM7", 115200)

mirobot1 = wlkatapython.Wlkata_UART()
mirobot1.init(serial1, -1)


def custom_tool_offset(mirobot, x, y, z):
    msg = (
        '$46=' + str(x) + '\n'
        '$47=' + str(y) + '\n'
        '$48=' + str(z)
    )

    mirobot.sendMsg(msg)


time.sleep(1)


print("\n[1] Offset = (0, 0, 0)")

custom_tool_offset(mirobot1, 0, 0, 0)

time.sleep(1)

coord_before = [mirobot1.getcoordinate(i) for i in range(1, 7)]

print("Offset 적용 전 좌표:")
print(coord_before)


print("\n[2] Offset = (0, 0, 120)")

custom_tool_offset(mirobot1, 0, 0, 120)

time.sleep(1)

coord_after = [mirobot1.getcoordinate(i) for i in range(1, 7)]

print("Offset 적용 후 좌표:")
print(coord_after)


print("적용 전:", coord_before)
print("적용 후:", coord_after)

serial1.close()