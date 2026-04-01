import torch
import torchvision
import torch.nn as nn
import cv2
import random
import json
from typing import Literal
from pydantic import BaseModel
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from  torchvision import models 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from deep_peep_snooker.utils.common import NumpyImage
from deep_peep_snooker.utils.playfield_finder import PlayfieldFinder




def prepare_img_for_resnet(img: np.ndarray):
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    img /= 255
    img_tensor = torch.tensor(img).permute(2, 0, 1).float()# .unsqueeze(0)
    return torchvision.transforms.functional.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def load_resnet18_model(path, device):
    model = models.resnet18()

    model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )

    state = torch.load(path)
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()

    return model

def put_text_on_img(img, text, color):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    text_color_bgr = (0, 255, 0)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            

    cv2.rectangle(img, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (15, 10 + text_height), font, font_scale, color, thickness)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = load_resnet18_model('models/shot-classifier-best.pt', device)


cap = cv2.VideoCapture('Recording 2026-04-01 184225.mp4')
counter = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print('error')
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = prepare_img_for_resnet(frame)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        y_hat = model(img)
        probs = F.sigmoid(y_hat).item()


    if probs >= 0.5:
        put_text_on_img(frame, 'OK', (0, 255, 0))
        # cv2.imshow(frame)

        frame = NumpyImage(frame)
        finder = PlayfieldFinder(frame)

        playfield = finder.detect_inner_playfield()

        frame = playfield.display_on_image(frame)
        
    else:
        put_text_on_img(frame, 'Wrong', (255, 0, 0))


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('video frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


