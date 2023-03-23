import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms, datasets

import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
import matplotlib.pyplot as plt

#import cv2
import time
import os
import numpy as np
from bs4 import BeautifulSoup

#from IPython import display
#import ultralytics
#from ultralytics import YOLO

import urllib.request
import zipfile
import io
import pandas as pd



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
    # URL of the ZIP file
    #zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip'
    #zip_url = 'https://drive.google.com/drive/shared-with-me/ENEL645/Fire+Data/fire.zip'

    # Download the ZIP file
    #response = urllib.request.urlopen(zip_url)

    # Extract the contents of the ZIP file to memory
    #zip_contents = io.BytesIO(response.read())
    #print(zip_contents)
    #zip_file = zipfile.ZipFile(zip_contents)
    zip_file = zipfile.ZipFile('./datasets/fire.zip', 'r')
    zip_file.extractall('./datasets')
    print("Dataset Extraction Complete.")
    #print(zip_file.namelist())
    #zip_file.printdir()
    
    ## Get all the xml label files into a list
    xml_label_path = zip_file.namelist()
    list_of_xml_label_files = [os.path.join('./datasets/',file) for file in xml_label_path if file.endswith('.xml')]
    #(list_of_xml_label_files)
    #print(len(list_of_xml_label_files))
    
    ## Read the bounding box data from the xml files, normalize it, and rewrite as txt files

    text_labels_path = "./datasets/fire/labels/text_labels"
    if not os.path.exists(text_labels_path):
        os.mkdir(text_labels_path)
        print("Text Labels Folder Created.")

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

def preprocess():

    """
    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = np.max([w, h])
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, hp, vp)
            return F.pad(image, padding, 0, 'constant')
    """

    image_size = 1280
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(1280),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    path = "./datasets/fire/"
    images = datasets.ImageFolder(root=path, transform=transform)

    print(len(images))

    #transform = T.ToPILImage()
    #img = transform(images[0][0])
    #img.save("test.jpg")

    #batch_t = torch.unsqueeze(img_t, 0)
    #print(batch_t.shape)

def train_yolo():
    model = YOLO("./runs/detect/yolov8n_50e2/weights/best.pt")

    #print(type(model.model)) # <class 'ultralytics.nn.tasks.DetectionModel'>
    #print(model.model) # Print model summary

    print("Starting training...")

    results = model.train(
        data='./datasets/fire/fire.yaml',
        imgsz=1280,
        epochs=75,
        batch=8,
        name='yolov8n_4'
    )

    print("Training finished!")

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

    # Step 1: Extract the downloaded dataset to prepare the data and labels
    #rewrite_xml_to_txt()
    #dataset_extraction_and_label_correction()

    #Step 2: Split the data into 3 datasets - Training, Validation and Testing
    train_valid_test_split()

    #preprocess()
    #train_yolo()
    #test_yolo()