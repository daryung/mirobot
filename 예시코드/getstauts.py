import wlkatapython
import serial
'' 'Get the robotic arm status' ''
serial1 = serial. Serial ("COM3", 38400) # Set the serial port and baud rate
mirobot1 =wlkatapython.Wlkata_UART () # Create a new mirobot1 object
mirobot1.init (serial1,1) # Set the robotic arm address

print (mirobot1.getStatus()) # Get the robotic arm status
serial1.close() # Close the serial port
