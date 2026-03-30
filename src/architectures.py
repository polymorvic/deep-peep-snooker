import torch
from torch import nn
from  torchvision import models 


def build_shot_classification_architecture() -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )

    return model


def load_shot_classification_weights(path: str, device: str | None = None) -> nn.Module:
    model = build_shot_classification_architecture()
    model.state_dict(torch.load(path, weights_only=True))
    return model.to(device) if device else model
