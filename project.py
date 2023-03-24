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

#from IPython import display
import ultralytics
from ultralytics import YOLO

#import urllib.request
import zipfile
import io
#import pandas as pd
import random
import shutil

def rewrite_xml_to_txt():

    # Get all the xml label files into a list
    og_label_path = "./datasets/fire/labels"
    list_of_xml_label_files = [os.path.join(og_label_path, file)
    for file in os.listdir(og_label_path)
        if file.endswith('.xml')]

    # Read the bounding box data from the xml files, normalize it, and rewrite as txt files

    text_labels_path = og_label_path + "/text_labels"
    if not os.path.exists(text_labels_path):
        os.mkdir(text_labels_path)

    for xml_file in list_of_xml_label_files:
        data = open(xml_file, 'r').read()
        bs_data = BeautifulSoup(data, "xml")
        bs_bndbox = bs_data.find_all('bndbox')

        filename = bs_data.find('filename').string
        #print(filename)
        width = bs_data.find('width').string
        #print(width)
        height = bs_data.find('height').string
        #print(height)

        boxes = list()
        for box in bs_bndbox:
            xmin = box.find('xmin').string
            xmin = float(xmin) / float(width)

            ymin = box.find('ymin').string
            ymin = float(ymin) / float(height)

            xmax = box.find('xmax').string
            xmax = float(xmax) / float(width)

            ymax = box.find('ymax').string
            ymax = float(ymax) / float(height)
            
            boxes.append("0 " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax))
    
        #text_file_path = path + "/" + filename[:-4] + ".txt"
        text_file_path = text_labels_path + "/" + filename[:-4] + ".txt"
        out = open(text_file_path, "w")

        for box in boxes:
            out.write(box + "\n")

def dataset_extraction_and_label_correction():

    # Extract the zipped dataset into local computer
    zip_file = zipfile.ZipFile('./datasets/fire.zip', 'r')    
    if not os.path.exists('./datasets/fire'):
        zip_file.extractall('./datasets')
    #print("Dataset Extraction Complete.")
    #print(zip_file,"\n")
    #print(zip_file.namelist())
    #zip_file.printdir()
    
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

def preprocess():

    # Path to images in dataset
    image_path = "./datasets/fire/images/"
    names = [file for file in os.listdir(image_path)]

    # Define transforms for data
    image_size = 1280
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(1280)
    ])

    # Transform and save the transformed images back into the dataset
    print("Image Transformation under progress ...")
    for file in names:
        img = Image.open(image_path + file)
        img_t = transform(img)
        img_t.save(image_path + file)
    print("Image Transformation Complete")

def train_yolo():
    
    # For the first run use the below weights to implement Transfer Learning
    #model = YOLO("yolov8n.pt")

    # All subsequent runs would be using he last best model
    model = YOLO("./runs/detect/yolov8n_train3_epoch100_batchsize8/weights/best.pt")
        
    #print(type(model.model)) # <class 'ultralytics.nn.tasks.DetectionModel'>
    #print(model.model) # Print model summary

    print("Starting training...")

    results = model.train(
        data='./datasets/fire/fire.yaml',
        imgsz=1280,
        epochs=1,
        batch=16,
        name='yolov8n_train4_epoch1_batchsize16'
    )

    print("Training finished!")
    # Output models are saved in ./runs/detect/<name>/weights/<best.pt/last.pt>

def test_yolo():
    
    print("Starting prediction...")

    model = YOLO("./runs/detect/yolov8n_50e/weights/best.pt")

    img_path = "./datasets/fire/test/images/large_(106).jpg"
    img = cv2.imread(img_path)
    results = model.predict(img_path)

    for i in range(len(results[0].boxes.xyxy)):
        p0 = (int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1]))
        p1 = (int(results[0].boxes.xyxy[i][2]), int(results[0].boxes.xyxy[i][3]))
        cv2.rectangle(img, p0, p1, color=(0), thickness=3)

    print("Saving output image.")
    cv2.imwrite('test.jpg', img)

    print("Classes found:")
    for r in results:
        for c in r.boxes.cls:
            print(model.names[int(c)])

    print("Prediction finished!")

if __name__ == '__main__':

    print("This is the MAIN")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)

    ########################################################################################################
    ###SECTION A: This is used once for extraction, pre-processing, splitting and dataloading of the dataset
    ########################################################################################################

    # The original dataset is here: https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset
    # The custom dataset is here: https://drive.google.com/drive/folders/1Zxwx6fIBil1rG_vFBmO8D7_7j_YATa9f

    # Step 1: Extract the downloaded dataset to prepare the data and corresponding labels
    #rewrite_xml_to_txt() #Donot use
    #dataset_extraction_and_label_correction()

    # Step 2: Preprocess the images by resizing and center cropping
    #preprocess()

    # Step 3: Split the data into 3 datasets - Training, Validation and Testing
    #train_valid_test_split() #Donot use
    #train_valid_test_stratified_split()

    #############################################################
    ###SECTION B: This is used for Training and Testing the model
    #############################################################

    train_yolo()
    #test_yolo()