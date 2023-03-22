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

def rewrite_xml_to_txt():

    # Get all the xml label files in the dataset
    path = "./datasets/fire/labels"
    labels = [os.path.join(path, file)
              for file in os.listdir(path)
              if file.endswith('.xml')]
    
    # Read the bounding box data from the xml files, normalize it, and rewrite as txt files
    for file in labels:
        data = open(file, 'r').read()
        bs_data = BeautifulSoup(data, "xml")
        bs_bndbox = bs_data.find_all('bndbox')

        name = bs_data.find('filename').string

        width = bs_data.find('width').string
        height = bs_data.find('height').string

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

        path2 = path + "/" + name[:-4] + ".txt"
        out = open(path2, "w")

        for box in boxes:
            out.write(box + "\n")

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
    model = YOLO("./runs/detect/yolov8n_4/weights/best.pt")

    #print(type(model.model)) # <class 'ultralytics.nn.tasks.DetectionModel'>
    #print(model.model) # Print model summary

    print("Starting training...")

    results = model.train(
        data='./datasets/fire/fire.yaml',
        imgsz=1280,
        epochs=75,
        batch=8,
        name='yolov8n_5'
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

    #rewrite_xml_to_txt()
    #train_valid_test_split()
    #preprocess()
    train_yolo()
    #test_yolo()