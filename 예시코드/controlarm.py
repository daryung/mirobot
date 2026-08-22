import wlkatapython
import serial
import time

serial1 = serial.Serial("COM7", 115200)
mirobot1 = wlkatapython.Wlkata_UART()
mirobot1.init(serial1, -1)
mirobot1.speed(2000)

mirobot1.writecoordinate(1,0,300,0,50,0,0,0)
mirobot1.writecoordinate(1,0,300,40,60,0,0,0)
mirobot1.writecoordinate(1,0,300,40,100,0,0,0)
mirobot1.writecoordinate(1,0,300,0,50,0,0,0)

#mirobot1.writeangle(0,0,20,0,0,0,0) #Absolute value
#mirobot1.writeangle(1,0,-10,0,0,0,0) #Incremental value