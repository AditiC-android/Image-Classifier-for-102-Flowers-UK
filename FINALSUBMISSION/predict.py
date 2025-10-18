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
import json

import argparse
import data

ap = argparse.ArgumentParser(description='This predict file will predict the flower and provide probability')
ap.add_argument('input_img', default='paind-project/flowers/test/1/image_06760.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', dest="gpu", action = "store", default="gpu")

args = ap.parse_args()
image_path = args.input_img
top_output = args.top_k
power = args.gpu
path = args.checkpoint
names = args.category_names

def get_the_names(names):
    with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    return cat_to_name
    
training_loader, testing_loader, validation_loader = data.augment()
model = data.load_model(path)

img_tensor = process_image(image_path)

probs, classes = data.predict(image_path, model, top_output, img_tensor, power)

##Will print these results here

j = 0
print("The Top 5 Classes Are.....")
while j<top_output:
    cat_to_name = get_the_names(names)
    print("{} with a probability of {}".format(cat_to_name[classes[i]], probs[i])
    j+=1
    


