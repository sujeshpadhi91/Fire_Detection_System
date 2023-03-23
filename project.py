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


# --------------------------- THESE DON'T NEED TO BE RUN AGAIN ----------------
# Used to test bounding boxes
def test_box():

    # Get an image
    img_path = "./special2.jpg"
    img = cv2.imread(img_path)

    # Calc box coords
    center_p = (int(512 * 0.607421875), int(512 * 0.548828125))
    w_h = (int(512 * 0.1171875),int(512 * 0.11328125))
    p0 = (int(center_p[0] - 0.5 * w_h[0]), int(center_p[1] - 0.5 * w_h[1]))
    p1 = (int(center_p[0] + 0.5 * w_h[0]), int(center_p[1] + 0.5 * w_h[1]))

    # Print coords
    print(p0)
    print(p1)
    
    # Draw coords on image
    cv2.rectangle(img, p0, p1, color=(0), thickness=3)

    # Save copy of image with box on it for verifying box is correct
    cv2.imwrite('test2.jpg', img)

# Used to rewrite the original XML labels into txt files for YOLOv8 model
def rewrite_xml_to_txt():

    # Get all the xml label files in the dataset
    path = "./datasets/fire/labels"
    labels = [os.path.join(path, file)
              for file in os.listdir(path)
              if file.endswith('.xml')]
    
    # Read the bounding box data from the xml files, normalize it, and rewrite as txt files
    for file in labels:
        # Open the file and parse the xml data
        data = open(file, 'r').read()
        bs_data = BeautifulSoup(data, "xml")

        # Save the file name
        name = bs_data.find('filename').string

        # Save the image width, height
        width = bs_data.find('width').string
        height = bs_data.find('height').string

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
def train_valid_test_split():

    # Get images dataset filepath
    path = "./datasets/fire/images/"

    # Get list of images in dataset
    names = [file for file in os.listdir(path)]

    # Split dataset into test, valid, and train subsets
    generator = torch.Generator().manual_seed(0)
    subsets = torch.utils.data.random_split(dataset=names, lengths=[0.1, 0.1, 0.8], generator=generator)

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


# ------------------------------------- TRAIN THE MODEL -----------------------
def train_yolo():

    # Load current best model (or just ./yolov8n.pt for a fresh version)
    model = YOLO("./runs/detect/yolov8n_4/weights/best.pt")

    # Print model summary (if interested - its long)
    #print(model.model)

    # Pass training data to model
    results = model.train(
        data='./datasets/fire/fire.yaml',   # Tells the model where to find the images/labels
        imgsz=1280,                         # Size of images
        epochs=1,                           # Number of epochs to train
        batch=8,                            # Number of images / batch
        name='yolov8n_test'                 # Output directory name - ./runs/detect/<name>
    )

    # Output models are saved in ./runs/detect/<name>/weights/<best.pt/last.pt>

# ------------------------------------- TEST THE MODEL ------------------------
# NOT FINISHED - ONLY DOES 1 IMAGE INSTEAD OF ALL TEST IMAGES
def test_yolo():
    
    # Load current best model
    model = YOLO("./runs/detect/yolov8n_50e/weights/best.pt")

    # Load a test image from custom dataset
    img_path = "./datasets/fire/test/images/large_(106).jpg"
    img = cv2.imread(img_path)

    # Use model to predict fire position in image (if any)
    results = model.predict(img_path)

    # Get the bounding box data from the results and draw it onto an image
    for i in range(len(results[0].boxes.xyxy)):
        p0 = (int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1]))
        p1 = (int(results[0].boxes.xyxy[i][2]), int(results[0].boxes.xyxy[i][3]))
        cv2.rectangle(img, p0, p1, color=(0), thickness=3)

    # Save a copy of the results
    cv2.imwrite('test.jpg', img)

if __name__ == '__main__':

    # THESE DON'T NEED TO BE RUN AGAIN
    # They only need to be ran once to create the custom dataset
    # The original dataset is here: https://github.com/siyuanwu/DFS-FIRE-SMOKE-Dataset
    # The custom dataset is here: https://drive.google.com/drive/folders/1Zxwx6fIBil1rG_vFBmO8D7_7j_YATa9f
    #test_box()
    #rewrite_xml_to_txt()
    #preprocess()
    #train_valid_test_split()

    # THESE ARE USED FOR TRAINING/TESTING THE MODEL
    train_yolo()
    #test_yolo()