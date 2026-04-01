import random
import torch
from torch import nn
import torch.nn.functional as F 
from pathlib import Path
import numpy as np
from pydantic import BaseModel
from typing import Literal
from torch.utils.data import Dataset, DataLoader


class ClassificationHistory(BaseModel):
    # zaimplementowac wiele klas + inne metryki do analizy - precision recall f1
    epoch_losses: list[float] = []
    epoch_accuracy: list[float] = []
    step_losses: list[float] = []
    best_loss: float = np.inf
    best_accuracy: float = 0
    _running_loss: float
    _correct_pred: float

    def on_epoch_start(self) -> None:
        self._running_loss = .0
        self._correct_pred = .0

    def on_epoch_end(self, dataset: Dataset, focus: Literal['loss', 'accuracy']) -> bool:
        """
        Czy dana epoka polepszyla model wzgledem treningu na podstawie metryki przekazywanej w argumencie focus
        """
        current_loss = self._running_loss / len(dataset)
        current_accuracy = self._correct_pred / len(dataset)
        self.epoch_losses.append(current_loss)
        self.epoch_accuracy.append(current_accuracy)

        if is_better_loss := current_loss < self.best_loss:
            self.best_loss = current_loss
  
        if is_better_accuracy := current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy

        return  {'loss': is_better_loss, 'accuracy': is_better_accuracy}[focus]

        
    def on_step_end(self, loss: float, y_hat: torch.Tensor, y_gt: torch.Tensor) -> None:
        y_pred = (F.sigmoid(y_hat).reshape(-1) >= 0.5).long()
        self._correct_pred += (y_pred == y_gt).sum().item()
        self._running_loss += loss
        self.step_losses.append(loss)

        

    def get_latest(self, mode: Literal['train', 'val']) -> str:
        return f"{mode} loss= {self.epoch_losses[-1]:.4f} {mode} accuracy= {self.epoch_accuracy[-1]:.4f}"
    

def run_training_loop(
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        learning_rate: float, 
        epochs_num: int, 
        scheduler_patience: int,
        model_output_path: Path,
        model_name: str,
        focus: Literal['loss', 'accuracy'],
        random_state: int | None = 123):


    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #  0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=scheduler_patience)
    focus = "loss"
    model = model.to(device)

    train_hist = ClassificationHistory()
    val_hist = ClassificationHistory()
    learning_rates = []

    for ep in range(epochs_num):
        model.train()

        train_hist.on_epoch_start()

        for batch_x, batch_y in train_dataloader:
            optimizer.zero_grad()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  
            y_hat = model(batch_x)


            loss = criterion(y_hat, batch_y.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            train_hist.on_step_end(loss.item(), y_hat, batch_y)

        train_hist.on_epoch_end(train_dataloader.dataset, focus)
        model.eval()
        val_hist.on_epoch_start()
        with torch.no_grad():

            for batch_x, batch_y in val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                y_hat = model(batch_x)

                loss = criterion(y_hat, batch_y.reshape(-1, 1))
                val_hist.on_step_end(loss.item(), y_hat, batch_y)


        save_best_model = val_hist.on_epoch_end(val_dataloader.dataset, focus)
        scheduler.step(val_hist.epoch_losses[-1])
        learning_rates.append(optimizer.param_groups[0]['lr'])

        print(f"Epoka {ep+1}/{epochs_num}: {train_hist.get_latest(mode='train')}, {val_hist.get_latest('val')}")

        checkpoint_data = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": ep + 1}
        torch.save(checkpoint_data, model_output_path / f'{model_name}-last.pt')
        if save_best_model:
            torch.save(checkpoint_data, model_output_path / f'{model_name}-best.pt')

