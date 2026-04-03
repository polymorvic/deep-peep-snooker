import torch
from torch import nn
import torch.nn.functional as F
from deep_peep_snooker.utils.common import ArrayLike

def predict_shot(model: nn.Module, img: ArrayLike) -> float:
    model.eval()
    with torch.no_grad():
        y_hat = model(img)
        prob = F.sigmoid(y_hat)
    return prob