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

def pretrained():
    alexnet = models.alexnet(weights=True)

    #print(alexnet)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open("dog.jpg")
    img_t = transform(img)
    print(img_t.shape)
    batch_t = torch.unsqueeze(img_t, 0)
    print(img_t.shape)

    alexnet.eval()

    out = alexnet(batch_t)
    print(out.shape)

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100


    print(index, percentage[index])

    resnet = models.resnet101(pretrained=True)

    resnet.eval()

    out = resnet(batch_t)

    _, index = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(print(percentage[idx].item())) for idx in index[0][:5]]

    print(dir(models))

def transfer_learning():

    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    train_dir = './data/train'
    valid_dir = './data/valid'
    test_dir = './data/test'

    bs = 32

    num_classes = 10

    data = {
        'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=test_dir, transform=image_transforms['test'])
    }

    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])
    test_data_size = len(data['test'])

    train_data = torch.utils.data.DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data = torch.utils.data.DataLoader(data['valid'], batch_size=bs, shuffle=True)
    test_data = torch.utils.data.DataLoader(data['test'], batch_size=bs, shuffle=True)

    print(train_data_size, valid_data_size, test_data_size)

    resnet50 = models.resnet50(pretrained=True)

    for param in resnet50.parameters():
        param.requires_grad = False

    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    )

    resnet50 = resnet50.to("cpu")

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters())

    epochs = 1

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        resnet50.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        history = list()

        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")

            optimizer.zero_grad()

            outputs = resnet50(inputs)

            print(outputs.shape)
            print(labels.shape)

            loss = loss_func(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy:{:.4f}".format(i, loss.item(), acc.item()))

        with torch.no_grad():
            resnet50.eval()

            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to("cpu")
                labels = labels.to("cpu")

                outputs = resnet50(inputs)

                loss = loss_func(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)
                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / float(train_data_size)

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / float(valid_data_size)

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))

        transform = image_transforms['test']
        test_image = Image.open("./data/test/giraffe/084_0072.jpg")
        plt.imshow(test_image)

        test_image_tensor = transform(test_image)

        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

        with torch.no_grad():
            resnet50.eval()

            out = resnet50(test_image_tensor)
            ps = torch.exp(out)

            topk, topclass = ps.topk(1, dim=1)
            print(topk, topclass)

def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])

def object_detection():

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
    def get_prediction(model, img_path, threshold):
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = model([img])
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        return pred_boxes, pred_class

    def object_detection_api(model, img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):

        boxes, pred_cls = get_prediction(model, img_path, threshold)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(len(boxes)):

            cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])), color=(0), thickness=text_th)

            cv2.putText(img, pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, color=(0), thickness=text_th)
        
        plt.figure(figsize=(20, 30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    object_detection_api(model, './data/people2.jpg', threshold=0.8)