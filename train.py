#!/usr/bin/env python
# coding: utf-8

import torch
import torchaudio
import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, path: Path = Path(r'D:\000data\ESC-50-master'),
                sample_rate: int = 8000,
                folds = [0]):
        self.path = path
        csv = pd.read_csv(path / Path('meta/esc50.csv'))
        self.csv = csv[csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)
        self.mels = x = torchaudio.transforms.MelSpectrogram(sample_rate=44100)
        self.amp = torchaudio.transforms.AmplitudeToDB()
        
        
    def __getitem__(self, index):
        x, _ = torchaudio.load(self.path / 'audio' / self.csv.iloc[index, 0], normalize=True)
        
        x = self.resample(x)
        x = self.mels(x)
        x = self.amp(x)
        
        y = self.csv.iloc[index, 2]
        
        return x, y
        
    def __len__(self):
        return len(self.csv)


class AudioNet(pl.LightningModule):
    
    def __init__(self, n_classes = 50, base_filters = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(base_filters, base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 2)
        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(base_filters * 4, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train():
    audionet = AudioNet()

    train_data = ESC50Dataset(folds=[1,2,3])
    val_data = ESC50Dataset(folds=[4])
    test_data = ESC50Dataset(folds=[5])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, num_workers=0)

    pl.seed_everything(0)
    trainer = pl.Trainer(gpus=1, max_epochs=1)
    trainer.fit(audionet, train_loader, val_loader)
    trainer.test(audionet, test_loader)


if __name__ == '__main__':
    train()









