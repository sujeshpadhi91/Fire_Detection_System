#................................CLIENT................................
import socket
import os

print("# Create a socket object")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("# Connect the socket to the server's address and port")
server_address = ('192.168.122.103', 12345)  # Replace with server's IP address
print("Sending connection request to: ", server_address)
client_socket.connect(server_address)

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Combine the script directory with your relative path
relative_path = 'images/client/input/image_file.jpg'
imagefile_path = os.path.join(script_directory, relative_path)

#imagefile_path = '/root/Fire_Detection_System/images/client/input/image_file.jpg'

with open(imagefile_path, 'rb') as file:
    while True:
        data = file.read(1024)
        if not data:
            break
        client_socket.send(data)
print("The image", imagefile_path, "was sent successfully")

print("Closing the connection.")
client_socket.close()
