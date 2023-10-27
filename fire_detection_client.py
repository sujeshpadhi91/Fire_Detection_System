#................................CLIENT................................
import socket
import os

print("# Create a socket object")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("# Connect the socket to the server's address and port")
#server_address = ('192.168.122.103', 12345)  # Replace with server's IP address
server_address = ('10.0.0.19', 12345)  # Replace with local server's IP address
print("Sending connection request to: ", server_address)
client_socket.connect(server_address)

# Get the directory where the script is located
#script_directory = os.path.dirname(os.path.abspath(__file__))
script_directory = 'C:/Users/sujes/FDS_repo/Fire_Detection_System'

# Combine the script directory with your relative path
client_input_relative_path = 'images/client/input/image_file.jpg'
imagefile_path = os.path.join(script_directory, client_input_relative_path)

#imagefile_path = '/root/Fire_Detection_System/images/client/input/image_file.jpg'

with open(imagefile_path, 'rb') as file:
    data = file.read(1024)
    while data:
        client_socket.send(data)
        data = file.read(1024)
print("The image", imagefile_path, "was sent successfully")

# Signal the end of the data transfer.
client_socket.shutdown(socket.SHUT_WR)

# Create the client output directory if not created
client_output_path = os.path.join(script_directory, 'images/client/output')
if not os.path.exists(client_output_path):
    os.mkdir(client_output_path)
    print("Client Output Directory Created.")

# Receive the image sent back by the server.
client_output_relative_path = 'images/client/output/received_processed_image_file.jpg'
processed_imagefile_to_return = os.path.join(script_directory, client_output_relative_path)
with open(processed_imagefile_to_return, "wb") as file:
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        file.write(data)
print("The processed image", processed_imagefile_to_return, "was received successfully")

print("Closing the connection.")
client_socket.close()
