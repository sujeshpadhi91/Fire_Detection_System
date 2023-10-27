import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms, datasets

import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import cv2
import time
import os
import numpy as np
from bs4 import BeautifulSoup

import ultralytics
from ultralytics import YOLO

import zipfile
import io
import random
import shutil
import socket

# Used to rewrite the original XML labels into txt files for YOLOv8 model
def rewrite_xml_to_txt():

    # Get all the xml label files into a list
    og_label_path = "./datasets/fire/labels"
    list_of_xml_label_files = [os.path.join(og_label_path, file)
    for file in os.listdir(og_label_path)
        if file.endswith('.xml')]

    text_labels_path = og_label_path + "/text_labels"
    if not os.path.exists(text_labels_path):
        os.mkdir(text_labels_path)

    # Read the bounding box data from the xml files, normalize it, and rewrite as txt files
    for xml_file in list_of_xml_label_files:
        data = open(xml_file, 'r').read()
        bs_data = BeautifulSoup(data, "xml")

        # Save the file name
        name = bs_data.find('filename').string

        # Save the image width, height
        width = bs_data.find('width').string
        #print(width)
        height = bs_data.find('height').string
        #print(height)

        # Find all objects labeled in image
        bs_obj = bs_data.find_all('object')

        # Loop through each object in the image
        boxes = list()
        for obj in bs_obj:

            # Check if the object is fire, if not skip it
            obj_name = obj.find('name').string
            if obj_name != 'fire':
                continue

            # Get the bounding box data for this object's label
            bs_box = obj.find_all('bndbox')
            for box in bs_box:
                xmin = box.find('xmin').string
                xmin = float(xmin) / float(width)   # Normalize

                ymin = box.find('ymin').string
                ymin = float(ymin) / float(height)  # Normalize

                xmax = box.find('xmax').string
                xmax = float(xmax) / float(width)   # Normalize

                ymax = box.find('ymax').string
                ymax = float(ymax) / float(height)  # Normalize

                center_x = (xmin + xmax) / 2.0      # Calc center of box x coord
                center_y = (ymin + ymax) / 2.0      # Calc center of box y coord

                box_width = xmax - xmin             # Calc box width
                box_height = ymax - ymin            # Calc box height

            # Add box label to list of box labels in the image
            boxes.append("0 " + str(center_x) + " " + str(center_y) + " " + str(box_width) + " " + str(box_height))

        # Get a path to save the text file output
        path2 = path + "/" + name[:-4] + ".txt"
        out = open(path2, "w")

        # Write all the box labels to the text file
        for box in boxes:
            out.write(box + "\n")

# Resize all images in dataset
def preprocess():

    # Path to images in dataset
    path = "./datasets/fire/images/"
    names = [file for file in os.listdir(path)]

    # Define transforms for data
    image_size = 1280
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(1280),
        #transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225]
        #)
    ])

    # Save the transformed images back into the dataset
    for file in names:
        img = Image.open(path + file)
        img_t = transform(img)
        img_t.save(path + file)

# Randomly assign image/label combos to train, test, valid sets and save them in the
#   appropriate location for the YOLOv8 model
def dataset_extraction_and_label_correction():

    # Extract the zipped dataset into local computer
    zip_file = zipfile.ZipFile('./datasets/fire.zip', 'r')    

    if not os.path.exists('./datasets/fire'):
        zip_file.extractall('./datasets')

    #print("Dataset Extraction Complete.")
    #print(zip_file,"\n")
    #print(zip_file.namelist())
    #zip_file.printdir()
    #C:\Users\sujes\ENEL645_git_repo\ENEL645A2\datasets\fire\labels\large_(1).xml
 
    # Get all the xml label files from the dataset into a list
    xml_label_path = zip_file.namelist()
    list_of_xml_label_files = [os.path.join('./datasets/',file) for file in xml_label_path if file.endswith('.xml')]
    #(list_of_xml_label_files)
    #print(len(list_of_xml_label_files))
    
    # Create the text_labels from xml
    text_labels_path = "./datasets/fire/labels/text_labels"
    if not os.path.exists(text_labels_path):
        os.mkdir(text_labels_path)
        print("Text Labels Folder Created.")

    # Read the bounding box data from the xml files, normalize it, and rewrite as txt files
    
    for xml_file in list_of_xml_label_files:
        # Open the file and parse the xml data
        data = open(xml_file, 'r').read()
        bs_data = BeautifulSoup(data, "xml")
        
        # Save the file name
        filename = bs_data.find('filename').string
        #print(filename)
        
        # Save the image width, height
        width = bs_data.find('width').string
        #print(width)
        height = bs_data.find('height').string
        #print(height)

        # Find all objects labeled in image
        bs_obj = bs_data.findAll('object')
        
        # Loop through each object in the image
        boxes = list()
        for obj in bs_obj:
            # Check if the object is fire, if not skip it
            obj_name = obj.find('name').string
            if obj_name != 'fire':
                continue
            #print(obj_name)

            # Get the bounding box data for this object's label
            bs_bndbox = bs_data.find_all('bndbox')
            for box in bs_bndbox:
                xmin = box.find('xmin').string
                xmin = float(xmin) / float(width)

                ymin = box.find('ymin').string
                ymin = float(ymin) / float(height)

                xmax = box.find('xmax').string
                xmax = float(xmax) / float(width)

                ymax = box.find('ymax').string
                ymax = float(ymax) / float(height)
                
                center_x = (xmin + xmax) / 2.0      # Calc center of box x coord
                center_y = (ymin + ymax) / 2.0      # Calc center of box y coord

                box_width = xmax - xmin             # Calc box width
                box_height = ymax - ymin            # Calc box height

            #boxes.append("0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax))
            boxes.append("0 " + str(center_x) + " " + str(center_y) + " " + str(box_width) + " " + str(box_height))
    
        # Get a path to save the text file output
        text_file_path = text_labels_path + "/" + filename[:-4] + ".txt"
        out = open(text_file_path, "w")
        
        # Write all the box labels to the text file
        for box in boxes:
            out.write(box + "\n")
    print("XML labels to Text labels Conversion Complete")

def train_valid_test_split():

    # Get images dataset filepath
    images_path = "./datasets/fire/images/"

    # Get list of images in dataset
    image_names = [file for file in os.listdir(images_path)]

    # Split dataset into test, valid, and train subsets
    generator = torch.Generator().manual_seed(0)
    #print("Generator\n",generator)
    subsets = torch.utils.data.random_split(dataset=image_names, lengths=[0.1, 0.1, 0.8], generator=generator)
    #print("\nSubsets Object\n",subsets)
    
    # Assign to test, valid, and train subsets
    test = subsets[0]
    valid = subsets[1]
    train = subsets[2]
    
    # Move the randomized subsets to their appropriate places
    for file in test:
        name = file[:-4]
        os.rename(path + file, "./datasets/fire/test/images/" + file)
        os.rename("./datasets/fire/labels/" + name + ".txt", "./datasets/fire/test/labels/" + name + ".txt")

    for file in valid:
        name = file[:-4]
        os.rename(path + file, "./datasets/fire/valid/images/" + file)
        os.rename("./datasets/fire/labels/" + name + ".txt", "./datasets/fire/valid/labels/" + name + ".txt")

    for file in train:
        name = file[:-4]
        os.rename(path + file, "./datasets/fire/train/images/" + file)
        os.rename("./datasets/fire/labels/" + name + ".txt", "./datasets/fire/train/labels/" + name + ".txt")

def train_valid_test_stratified_split():

    # Get images dataset filepath
    images_path = "./datasets/fire/images/"

    # Get list of images in dataset
    image_names = [file for file in os.listdir(images_path)]
    #print(image_names)
      
    # Using Stratified Sampling to split images into Training, Validation, and Testing subsets
    training_filenames = []
    validation_filenames = []
    testing_filenames = []

    total_image_count = len(image_names)
    random.shuffle(image_names)
    training_filenames.extend(image_names[:int(total_image_count*0.8)])
    validation_filenames.extend(image_names[int(total_image_count*0.8):int(total_image_count*0.9)])
    testing_filenames.extend(image_names[int(total_image_count*0.9):])
    #print(len(training_filenames),len(validation_filenames),len(testing_filenames))
    #print(training_filenames,validation_filenames,testing_filenames)
    
     #Create the Training - Validation - Testing dataset folders
    if not os.path.exists(images_path[:-8]+"/train"):
        os.mkdir(images_path[:-8]+"/train/")
        if not os.path.exists(images_path[:-8]+"/train/images"):
            os.mkdir(images_path[:-8]+"/train/images")
            os.mkdir(images_path[:-8]+"/train/labels")
        print("Training Dataset Folder Created.")
    
    if not os.path.exists(images_path[:-8]+"/valid"):
        os.mkdir(images_path[:-8]+"/valid/")
        if not os.path.exists(images_path[:-8]+"/valid/images"):
            os.mkdir(images_path[:-8]+"/valid/images")
            os.mkdir(images_path[:-8]+"/valid/labels")
        print("Validation Dataset Folder Created.")

    if not os.path.exists(images_path[:-8]+"/test"):
        os.mkdir(images_path[:-8]+"/test/")
        if not os.path.exists(images_path[:-8]+"/test/images"):
            os.mkdir(images_path[:-8]+"/test/images")
            os.mkdir(images_path[:-8]+"/test/labels")
        print("Testing Dataset Folder Created.")
    
    # Create the respective data for Training - Validation - Testing in the dataset folders
    for file in training_filenames:
        name = file[:-4]
        #os.rename(images_path + file, "./datasets/fire/train/images/" + file)
        #os.rename("./datasets/fire/labels/text_labels/" + name + ".txt", "./datasets/fire/train/labels/" + name + ".txt")
        shutil.copy(images_path + file, "./datasets/fire/train/images/" + file)
        shutil.copy("./datasets/fire/labels/text_labels/" + name + ".txt", "./datasets/fire/train/labels/" + name + ".txt")

    for file in validation_filenames:
        name = file[:-4]
        #os.rename(images_path + file, "./datasets/fire/val/images/" + file)
        #os.rename("./datasets/fire/labels/text_labels/" + name + ".txt", "./datasets/fire/val/labels/" + name + ".txt")
        shutil.copy(images_path + file, "./datasets/fire/val/images/" + file)
        shutil.copy("./datasets/fire/labels/text_labels/" + name + ".txt", "./datasets/fire/val/labels/" + name + ".txt")

    for file in testing_filenames:
        name = file[:-4]
        #os.rename(images_path + file, "./datasets/fire/test/images/" + file)
        #os.rename("./datasets/fire/labels/text_labels/" + name + ".txt", "./datasets/fire/test/labels/" + name + ".txt")
        shutil.copy(images_path + file, "./datasets/fire/test/images/" + file)
        shutil.copy("./datasets/fire/labels/text_labels/" + name + ".txt", "./datasets/fire/test/labels/" + name + ".txt")

# ------------------------------------- TRAIN THE MODEL -----------------------
def train_yolo():


    # All subsequent runs would be using he last best model
    # Load current best model (or just ./yolov8n.pt for a fresh version)
    model = YOLO("./runs/detect/yolov8n_1/weights/best.pt")
    
    #print(type(model.model)) # <class 'ultralytics.nn.tasks.DetectionModel'>
    #print(model.model) # Print model summary


    # Print model summary (if interested - its long)
    #print(model.model)

    # Pass training data to model
    results = model.train(

        data='./datasets/fire/fire.yaml',   # Tells the model where to find the images/labels
        imgsz=1280,                         # Size of images
        epochs=100,                         # Number of epochs to train
        batch=8,                            # Number of images / batch
        name='yolov8n_2'                    # Output directory name - ./runs/detect/<name>
    )

    # Output models are saved in ./runs/detect/<name>/weights/<best.pt/last.pt>

# ------------------------------------- TEST THE MODEL ------------------------
def test_yolo():
    
    # Load current best model
    model = YOLO("./runs/detect/yolov8n_2/weights/best.pt")

    results = model.val(
        data='./datasets/fire/test.yaml',
        imgsz=1280,
        batch=8,
    )

    ## Load a test image from custom dataset
    #image_path = "./datasets/fire/test/images/middle_(4608).jpg"
    #image_path = ["./datasets/fire/test/images/middle_(4608).jpg"]
    
    for image in names:
        img = cv2.imread(image_path+image)
        
        # Use model to predict fire position in image (if any)
        results = model.predict(img)

        # Get the bounding box data from the results and draw it onto an image
        for i in range(len(results[0].boxes.xyxy)):
            p0 = (int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1]))
            p1 = (int(results[0].boxes.xyxy[i][2]), int(results[0].boxes.xyxy[i][3]))
            cv2.rectangle(img, p0, p1, color=(0,0,255), thickness=5) #color format = (B,G,R)

        # Save a copy of the results
        print("Saving output image.")
        tested_filename = ''.join('./predictions/tested_'+image)
        cv2.imwrite(tested_filename, img)

        print("Classes found:")
        for r in results:
            for c in r.boxes.cls:
                print(model.names[int(c)])
        

# ------------------------------------- USE THE MODEL ------------------------#
def predict(imagefile_received):

    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with your relative path
    relative_processedimage_path = 'images/server/output/fire.jpg'
    processedimagefile_path = os.path.join(script_directory, relative_processedimage_path)

    relative_model_path = 'runs/detect/yolov8n_2/weights/best.pt'
    model_path = os.path.join(script_directory, relative_model_path)
    
    # Load current best model
    #model = YOLO("C:/Users/sujes/FDS_repo/Fire_Detection_System/runs/detect/yolov8n_2/weights/best.pt")
    #model = YOLO("/root/Fire_Detection_System/runs/detect/yolov8n_2/weights/best.pt")
    model = YOLO(model_path)
    
    #results = model.predict('C:/Users/sujes/FDS_repo/Fire_Detection_System/images/input')
    results = model.predict(imagefile_received)
    
    transform = T.ToPILImage()

    # Iterate through all the results
    for result in results:
        boxes = result.boxes # Boxes object for bbox outputs
        probs = result.probs # Class probabilities for classification outputs
        img = transform(result.orig_img)

        res_plotted = result.plot()
        #cv2.imwrite('test.jpg', res_plotted)
        #cv2.imwrite('C:/Users/sujes/FDS_repo/Fire_Detection_System/images/server/output/fire.jpg', res_plotted)
        #cv2.imwrite('/root/Fire_Detection_System/images/server/output/fire.jpg', res_plotted)
        cv2.imwrite(processedimagefile_path, res_plotted)
        # break

        """
        # Iterate through every bounding box in the image
        for box in boxes:

            # Get bounding box coords
            p0 = print(box.xyxy[0][0])
            p1 = print(box.xyxy[0][1])
            p2 = print(box.xyxy[0][2])
            p3 = print(box.xyxy[0][3])

            # Draw bounding box onto image
            #cv2.rectangle(img, p0, p1, color=(0), thickness=3)


        # Display
        img.show()

        # Save
        #cv2.imwrite('test.jpg', img)
        """
        # CLI version:
        #yolo task=detect mode=predict model=runs/detect/yolov8n_2/weights/best.pt source='https://www.youtube.com/watch?v=yaZF4Hznalc' show=True imgsz=1280 name=yolov8n_v8_50e_infer1280 hide_labels=True

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    ###########################################################################################################
    ## SECTION A: This is used once for extraction, pre-processing, splitting and dataloading of the dataset ##
    ###########################################################################################################

    # The original dataset is here: https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset
    # The custom dataset is here: https://drive.google.com/drive/folders/1Zxwx6fIBil1rG_vFBmO8D7_7j_YATa9f

    # Step 1: Extract the downloaded dataset to prepare the data and corresponding labels
    #dataset_extraction_and_label_correction()

    # Step 2: Preprocess the images by resizing and center cropping
    #preprocess()

    # Step 3: Split the data into 3 datasets - Training, Validation and Testing
    #train_valid_test_stratified_split()

    ################################################################
    ## SECTION B: This is used for Training and Testing the model ##
    ################################################################

    #train_yolo()
    #test_yolo()
    #predict()

    #################################################################
    ## SECTION C: This is used for deploying the model on a server ##
    #################################################################

    print("# Create a socket object")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("# Bind the socket to a specific address and port")
    server_address = ('0.0.0.0', 12345)  # Listen on all available network interfaces
    server_socket.bind(server_address)

    print("# Listen for incoming connections")
    server_socket.listen(1)

    print("Waiting for a connection...")
    client_socket, client_address = server_socket.accept()

    print("Server Socket")
    print(server_socket)

    print("Connection established with the client:", client_address)
    
    #imagefile_to_receive = '/root/Fire_Detection_System/images/server/input/received_imagefile.jpg'
    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Create the server input and output directories if not created
    server_path = os.path.join(script_directory, 'images/server')
    if not os.path.exists(server_path):
        os.mkdir(server_path)
        print("Server Directory Created.")
    input_path = os.path.join(script_directory, 'images/server/input')
    if not os.path.exists(input_path):
        os.mkdir(input_path)
        print("Server Input Directory Created.")
    output_path = os.path.join(script_directory,'images/server/output')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("Server Output Directory Created.")

    #imagefile_to_receive = './images/server/input/received_imagefile.jpg'
    # Combine the script directory with your relative path
    server_input_relative_path = 'images/server/input/received_image_file.jpg'
    imagefile_to_receive = os.path.join(script_directory, server_input_relative_path)

    with open(imagefile_to_receive, 'wb') as file:
        data = client_socket.recv(1024)
        while data:
            file.write(data)
            data = client_socket.recv(1024)
    print("The image", imagefile_to_receive, "was received successfully")
    predict(imagefile_to_receive)
    print("The image", imagefile_to_receive, "was processed successfully")

    # Send the image back to the client.
    server_output_relative_path = 'images/server/output/fire.jpg'
    imagefile_to_return = os.path.join(script_directory, server_output_relative_path)    
    with open(imagefile_to_return, "rb") as file:
        data = file.read()
        while data:
            client_socket.send(data)
            data = file.read(1024)
    print("The processed image", imagefile_to_return, "was returned successfully")

    print("Closing the connection.")
    client_socket.close()
    server_socket.close()
