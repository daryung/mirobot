import wlkatapython
import serial


serial1 = serial. Serial ("COM3", 115200) # Set the serial port and baud rate
mirobot1 =wlkatapython. Wlkata_UART () # Create a new mirobot1 object
mirobot1.init (serial1,-1) # Set the robotic arm address
mirobot1.restart() # Restart multifunction controller
serial1.close() # Close the serial port
