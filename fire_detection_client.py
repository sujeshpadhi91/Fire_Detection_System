#................................CLIENT................................
import socket

print("# Create a socket object")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("# Connect the socket to the server's address and port")
server_address = ('10.0.0.19', 12345)  # Replace with server's IP address
print("Sending connection request to: ", server_address)
client_socket.connect(server_address)

imagefile_path = './images/client/input/image_file.jpg'

with open(imagefile_path, 'rb') as file:
    while True:
        data = file.read(1024)
        if not data:
            break
        client_socket.send(data)
print("The image", imagefile_path, "was sent successfully")

print("Closing the connection.")
client_socket.close()
