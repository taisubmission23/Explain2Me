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
from scipy import optimize
from scipy.optimize import brenth
import sys
import argparse
from scipy.ndimage.filters import gaussian_filter
import skimage.io
import skimage.filters
from random import randrange
from PIL import Image
import tensorflow as tf
from absl import app, flags
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from torchvision.io import read_image
import matplotlib.image as mpimg
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import csv
from tqdm import tqdm
#sys.path.append("../")
from xception.network.models import model_selection

def write_to_file(file_path, data):
    file = open(file_path, "a")
    file.write(data + "\n")
    file.close()

parser = argparse.ArgumentParser()

device = torch.device('cpu')

parser.add_argument('-mp', '--modelPath', default="/home/ccrum/Documents/NotreDame/Research/Code/Cyborg/Cyborg_weights/densenet_cyborg_1/Logs/final_model.pth", type=str)
parser.add_argument('-n', '--network', default="resnet", type=str)
parser.add_argument('-d', '--dataset', default="/home/ccrum/Documents/NotreDame/Research/Code/Cyborg/Data/test/1_fake/stylegan2-0.5_aligned/000000", type=str)
parser.add_argument('-t', '--type', default="salt_pepper")

args = vars(parser.parse_args())

# ## Loading in the weights
output_file = str(args['modelPath'])
output_file = output_file.replace('/', '-')
output_file = output_file[1:]
output_file = "./results/" + output_file + "-" + args["type"] + ".txt"

# Dataset
dataset_name = args["dataset"]
dataset_name = dataset_name.split("/")[7]


# Training type
model_training_name = args["modelPath"]
model_training_name = model_training_name.split("/")[6]

output_file = "./results/" + dataset_name + "-" + model_training_name + "-" + args["type"] + "-" + ".txt"

if "STYLEGAN1" in args["dataset"]:
    data_name = "STYLEGAN1"

if "STYLEGAN2" in args["dataset"]:
    data_name = "STYLEGAN2"

if "celeba-hq_real_aligned" in args["dataset"]:
    data_name = "celeba-hq_real_aligned"

if "ffhq_aligned" in args["dataset"]:
    data_name = "ffhq_aligned"

if "stargan_aligned" in args["dataset"]:
    data_name = "stargan_aligned"

if "progan_aligned" in args["dataset"]:
    data_name = "progan_aligned"

if "stylegan3-0.5_aligned" in args["dataset"]:
    data_name = "stylegan3-0.5_aligned"

if "stylegan-ada-0.5_aligned" in args["dataset"]:
    data_name = "stylegan-ada-0.5_aligned"


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

save_directory = "/scratch365/ccrum/Cyborg/camdeg-salt/" + data_name + "/" + args['type'] + "/"  + args["network"] + "/"  + model_training_name + "/"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for image_file in resample(glob.glob(directory), n_samples=10000, replace=True, random_state=1):
    # Read the image
    image = skimage.io.imread(image_file)
    image_name = os.path.basename(image_file)
    salt_pepper = skimage.util.random_noise(image, mode="s&p")
    salt_pepper = np.array((salt_pepper * 255).astype(np.uint8))
    salt_pepper = Image.fromarray(salt_pepper)

    save_dir = save_directory + image_name

    salt_pepper.save(save_dir)
    #print(type(salt_pepper))

    #skimage.io.imsave("S&P.png", salt_pepper)
    
    tranformImage = transform(salt_pepper)
    tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
    # tranformImage = torch.tensor(tranformImage,requires_grad=True)
    tranformImage = tranformImage.to(device)

    with torch.no_grad():
        output = model(tranformImage)

    PAScore = sigmoid(output).detach().cpu().numpy()[:, 1][0]
    data = str(image_name)  + "," + str(PAScore) + "," + str(model_training_name) + "," + str(args["network"]) + "," + str(dataset_name) + "," + str(args["type"])
    print(data)
    write_to_file(output_file, data)

