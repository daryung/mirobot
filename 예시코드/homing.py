
import wlkatapython
import serial

'' 'The mechanical arm returns to zero to' ''

serial1 = serial. Serial("COM3", 115200) # Set the serial port and baud rate
mirobot1 = wlkatapython.Wlkata_UART() # Create a new mirobot1 object
mirobot1.init(serial1,-1) 

mirobot1.homing() 
serial1.close() 
