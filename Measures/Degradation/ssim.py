# SSIM.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
#import seaborn as sns
from copy import deepcopy

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import glob
from sklearn.utils import resample
from scipy import optimize
from scipy.optimize import brenth
import sys
import argparse

import skimage.io
import matplotlib.pyplot as plt
import skimage.filters

from random import randrange
from PIL import Image

import os
import pandas as pd
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
from absl import app, flags
#from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from sklearn.utils import resample
from torchvision.io import read_image


#from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
#from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
#from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

#import cv2
import matplotlib.pyplot as plt
from PIL import Image as im
import matplotlib.image as mpimg
import glob

#from keras.applications.vgg19 import preprocess_input

#from tensorflow.python.ops.numpy_ops import np_config

import argparse

#np_config.enable_numpy_behavior()

import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
#sys.path.append("../")
#from xception.network.models import model_selection

from sklearn.utils import resample

from PIL import Image
import torchvision.transforms as transforms

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import GradCAM
from torchcam.methods import GradCAMpp


import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

from random import randrange
from skimage.metrics import structural_similarity as ssim


## This is used to calculate the blur amount based on a polynomial
def get_curve(max_blur,val,order=4):
    val = (1/(max_blur**(order-1)))*(val**order) # (1/max^3)x^4
    return val


def show_heatmap(img,cam):
    plt.cla()
    #cam *= (255.0/cam.max()) # Change to range 0 to 255
    cam = gaussian_filter(cam, sigma=5) # Blur boundary between different annotation density levels

    # Generate heatmap of saliency map
    hmax = sns.heatmap(cam,
        alpha = 0.5,
        zorder=2,
        edgecolor="none",
        linewidths=0.0,#,)
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        rasterized=True,
        cmap="jet")

    hmax.imshow(img,
          aspect = hmax.get_aspect(),
          extent = hmax.get_xlim() + hmax.get_ylim(),
          zorder = 1)
    hmax.axes.get_xaxis().set_visible(False)
    hmax.axes.get_yaxis().set_visible(False)
    plt.grid(False)
    plt.tight_layout(pad=0)
    plt.title("Heatmap representation of saliency map")
    plt.show()


def blur_image(im,cam,max_blur,curved):
    blurred_im = deepcopy(im) # to hold final image
    stacked_max = max(cam.max(),1) # Make sure there are no 0 maxes
    #cam *= (255.0/stacked_max) # transform to 0 to 255 range
    cam = scipy.ndimage.gaussian_filter(cam, 5) # Blur this image so no hard boundaries between different levels
    scaled_img = cam/(cam.max()/max_blur) # Scale image back to range 0 to sigma max so we get correct blur level

    scaled_img = np.around(scaled_img,decimals=1) # Round to one decimal place
    uniq_vals = np.unique(scaled_img)
    for scaled_val in uniq_vals: # for all possible blur levels
        blur_amount = round(max_blur - scaled_val, 1)
        if curved: # based on polynomial instead of linear
            blur_amount  = get_curve(max_blur,blur_amount,order=4)

        if blur_amount != 0: # if blurring required
            indices = np.where(scaled_img == scaled_val) # where this level of blur should be applied
            qwerty = scipy.ndimage.gaussian_filter(im, blur_amount)
            blurred_im[indices] = qwerty[indices] # apply that blur level to the specified pixels
    return blurred_im


def write_to_file(file_path, data):
    file = open(file_path, "a")
    file.write(data + "\n")
    file.close()


import itertools

def findsubsets(s, n):
        return list(itertools.combinations(s, n))

# Driver Code
s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
n = 2

subset = findsubsets(s, n)

parser = argparse.ArgumentParser()
device = torch.device('cpu')

parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-m', '--model', default="cyborg")

args = vars(parser.parse_args())

if args["model"] == "cyborg":
        #model = "/resnet/resnet_cyborg_1"
        model = "/densenet/densenet_cyborg_1"

elif args["model"] == "xent":
        #model = "/resnet/resnet_xent_1"
        model = "/densenet/densenet_xent_1"

main_directory = "/home/ccrum/*"

dataset = args["dataset"]

#prefix_path = "/scratch365/ccrum/Cyborg/test-cams/gradcam/"
prefix_path = "/scratch365/ccrum/Cyborg/degrad-cams/gradcam/"
prefix_path_baseline = "/scratch365/ccrum/Cyborg/test-cams/gradcam/"
files_dict = {}

outfile = "./results/" + dataset + "-" + args["model"] + ".csv"

first_model = prefix_path + dataset + model + "/upscaled/background/*"
second_model = prefix_path_baseline + dataset + model + "/upscaled/"
#first_model_name = model + str(i[0])
#second_model_name = model + str(i[1])
#write_to_file("test.txt", first_model)
#write_to_file("test.txt", second_model)
for cam_file in glob.glob(first_model):
    #average_SSIM = []
    cam_1 = cv2.imread(cam_file, cv2.IMREAD_UNCHANGED)
    cam_name = os.path.basename(cam_file)
    cam_file_2 = second_model + cam_name
    cam_2 = cv2.imread(cam_file_2, cv2.IMREAD_UNCHANGED)
    try:
        ssim = skimage.metrics.structural_similarity(cam_1, cam_2)
    except:
        ssim = "NA"
    #write_to_file("test.txt", ssim)
    files_dict.setdefault(cam_name, []).append(ssim)
    #print(str(files_dict)):
    #write_to_file("test.txt", "Error")
pd.DataFrame.from_dict(data=files_dict, orient='index').to_csv(outfile, header=False)

