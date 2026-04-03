import torch
import torchvision
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torchvision import models


import numpy as np

from deep_peep_snooker.utils.common import NumpyImage
from deep_peep_snooker.utils.playfield_finder import PlayfieldFinder
from deep_peep_snooker.architectures import load_shot_classification_weights
from deep_peep_snooker.utils.helpers import create_device, preprare_single_frame_for_inference, put_text_on_img


device = create_device()
model = load_shot_classification_weights("models/shot-classifier-best.pt", device)

cap = cv2.VideoCapture("Recording 2026-04-01 184225.mp4")
counter = 0
playfield = None
has_moved_to_front_camera = False
while True:
    counter += 1
    ret, frame = cap.read()

    if not ret:
        print("the end")
        break

    frame, img = preprare_single_frame_for_inference(frame, device)

    with torch.no_grad():
        y_hat = model(img)
        probs = F.sigmoid(y_hat).item()

    if probs >= 0.5:
        put_text_on_img(frame, "OK", (0, 255, 0))

        if not has_moved_to_front_camera:
            has_moved_to_front_camera = True

            frame = NumpyImage(frame)
            finder = PlayfieldFinder(frame)

            try:
                playfield = finder.detect_inner_playfield()
            except ValueError as e:
                print(f"frame {counter}, {e}")

        if playfield is not None:
            frame = playfield.display_on_image(frame)

    else:
        has_moved_to_front_camera = False
        put_text_on_img(frame, "Wrong", (255, 0, 0))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("video frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
