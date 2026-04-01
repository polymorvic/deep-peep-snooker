import torch
import torchvision

from torch import nn

class MinMaxTransform(torch.nn.Module):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img.float().div(255.0)
    

def compose_transform(*ops: nn.Module) -> torchvision.transforms.Compose:

    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        MinMaxTransform(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        *ops
        ])
    

transform = compose_transform()