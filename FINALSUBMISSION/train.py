import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import data

ap = argparse.ArgumentParser(description='Train.py')

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="source", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_checkpoint", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.2)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="models", action="store", default="densenet121", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=256)

args = ap.parse_args()
path = pa.data_dir
path_two = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_units = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

trainloader, testloader, validloader = data.augment(path)

model, optimizer, criterion = data.prepare(structure,dropout,hidden_units,lr,power)

data.train(model, optimizer, criterion, epochs, 20, trainloader, validloader, power)
data.save_checkpoint(path_two, stucture, hidden_units, dropout, lr)



print("Training has completed")

