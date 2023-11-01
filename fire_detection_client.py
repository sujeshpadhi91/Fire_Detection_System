#................................CLIENT................................
import socket
import os

print("# Connect the socket to the server's address and port")
server_address = ('192.168.122.103', 12345)  # Replace with server's IP address
#server_address = ('10.13.76.150', 12345)  # Replace with local server's IP address
print("Sending connection request to: ", server_address)

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Combine the script directory with your input path
client_input_directory = os.path.join(script_directory, 'images/client_input')
client_input_file_list = os.listdir(client_input_directory)

# Create the client output directory if not created
client_output_directory = os.path.join(script_directory, 'images/client_output')
os.makedirs(client_output_directory, exist_ok=True)

for filename in client_input_file_list:    
    # Connect to the server.
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    print("Connected to: ", server_address)
    client_input_file_path = os.path.join(client_input_directory, filename)
    client_output_file_path = os.path.join(client_output_directory, filename)

    ######################################################
    # SENDING THE IMAGE FILE TO THE SERVER FOR PROCESSING
    ######################################################

    # Send the filename to the server
    client_socket.send(filename.encode())
    print("The filename:", filename, "was sent successfully")
    acknowledgment = client_socket.recv(1024)
    print("Server says:", acknowledgment.decode())
    
    # Send the file to the server
    with open(client_input_file_path, 'rb') as file:
        data = file.read(1024)
        while data:
            client_socket.send(data)
            data = file.read(1024)
    
    # Signal the end of the data transfer.
    client_socket.shutdown(socket.SHUT_WR)
    print("The image", filename, "was sent successfully from ", client_input_file_path)
    
    # Receive the acknowledgment message from the server
    acknowledgment = client_socket.recv(1024)
    print("Server says:", acknowledgment.decode())

    #############################################
    # RECEIVING FROM THE CLIENT AFTER PROCESSING
    #############################################

    # Receive the processed files sent back by the server.
    with open(client_output_file_path, "wb") as file:
        data = client_socket.recv(1024)
        while data:
            file.write(data)
            data = client_socket.recv(1024)
    print("The image", filename, "was processed and received successfully at ", client_output_file_path)

    print("Closing the connection.")
    client_socket.close()
