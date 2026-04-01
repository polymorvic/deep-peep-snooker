from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from deep_peep_snooker.ops import compose_transform
from torchvision.io import read_image

def run_shot_classification_inference(model: nn.Module, img_path: str | Path, device: str | None):
    model = model.to(device) if device else model
    model.eval()

    img = read_image(img_path)
    img = compose_transform(img)
    # img = img.unsqueeze(0)

    img = img.to(device) if device else img

    with torch.no_grad():
        y_hat = model(img)
        probs = F.sigmoid(y_hat)



# jak sonfigurowac pyproject toml zeby mozna bylo wewnatrz projektu nie robic from src.ops tylko from ops