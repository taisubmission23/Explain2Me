import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
from copy import deepcopy
import tensorflow as tf
import pandas as pd
import json
import glob
from sklearn.utils import resample
import sys
import argparse
import skimage.io
import skimage.filters
from random import randrange
from PIL import Image
from absl import app, flags
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import csv
from tqdm import tqdm
from torchvision.transforms.functional import normalize, resize, to_pil_image
from xception.network.models import model_selection

#sys.path.append("../")

# generic function to write results to file
def write_to_file(file_path, data):
    file = open(file_path, "a")
    file.write(data + "\n")
    file.close()

parser = argparse.ArgumentParser()

device = torch.device('cpu')

parser.add_argument('-mp', '--modelPath', default="/home/ccrum/Documents/NotreDame/Research/Code/Cyborg/Cyborg_weights/densenet_cyborg_1/Logs/final_model.pth", type=str)
parser.add_argument('-n', '--network', default="resnet", type=str)
parser.add_argument('-s', '--size', default=10000, type=int)
parser.add_argument('-d', '--dataset', default="/home/ccrum/Documents/NotreDame/Research/Code/Cyborg/Data/test/1_fake/stylegan2-0.5_aligned/000000", type=str)

args = vars(parser.parse_args())

directory = str(args['dataset'])
directory = directory + "/*"

sample_size = args['size']

# Dataset
dataset_name = args["dataset"]
dataset_name = dataset_name.split("/")[-1]

# Training type
model_training_name = args["modelPath"]
model_training_name = model_training_name.split("/")[-3]

# ## Loading in the weights
output_file = str(args['modelPath'])
output_file = output_file.replace('/', '-')
output_file = output_file[1:]
output_file = "./aug_3/" + model_training_name + "-" + dataset_name + ".txt"

# Load weights of single binary DesNet121 model
weights = torch.load(args["modelPath"], map_location=device)

if args["network"] == "resnet":
    im_size = 224
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
elif args["network"] == "inception":
    im_size = 299
    model = models.inception_v3(pretrained=True,aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
elif args["network"] == "xception":
    im_size = 299
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
else: # else DenseNet
    im_size = 224
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)

model.load_state_dict(weights['state_dict'])
model = model.to(device)
model.eval()

if args["network"] == "xception":
    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([im_size, im_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
else:
    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([im_size, im_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


sigmoid = nn.Sigmoid()

directory = str(args['dataset'])
directory = directory + "/*"

for image_file in resample(glob.glob(directory), n_samples=sample_size, replace=True, random_state=1):
    # Read the image
    image = Image.open(image_file).convert('RGB')
    image_name = os.path.basename(image_file)
    
    # Image transformation
    tranformImage = transform(image)
    image.close()
    tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
    tranformImage = tranformImage.to(device)

    # Get prediction on original input image
    with torch.no_grad():
        output = model(tranformImage)

    PAScore = sigmoid(output).detach().cpu().numpy()[:, 1][0]
    data = str(image_name) + "," + str(PAScore) + "," + str(model_training_name) + "," + str(dataset_name)
    print(data)
    write_to_file(output_file, data)
