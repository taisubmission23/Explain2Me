# Grad-Cam Generation
# 8/2/22

# Import Libraries
from configparser import Interpolation
import numpy as np
import os
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import csv
import argparse
from tqdm import tqdm
sys.path.append("../")
from xception.network.models import model_selection
import torchvision.transforms as transforms
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import GradCAM
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
import torchvision
import pandas as pd
from sklearn.utils import resample

def write_to_file(file_path, data):
        file = open(file_path, "a")
        file.write(data + "\n")
        file.close()

# Argparse
parser = argparse.ArgumentParser()

device = torch.device('cpu')
parser.add_argument('-d','--dataset', default="/home/ccrum/Documents/NotreDame/Research/Code/Cyborg/Crane-Delierables/Grad-Complexity/augmented_images", type=str)
parser.add_argument('-mp', '--modelPath', default="/home/ccrum/Documents/NotreDame/Research/Code/Cyborg/Cyborg_weights/densenet_cyborg_1/Logs/final_model.pth" ,type=str)
parser.add_argument('-n','--network', default="resnet", type=str)
parser.add_argument('-gt', '--gradcamType', default="gradcam", type=str)
parser.add_argument('-s', '--size', default="upscaled", type=str)

#args = parser.parse_args([])

args = vars(parser.parse_args())

# Training type
model_training_name = args["modelPath"]
model_training_name = model_training_name.split("/")[6]

if "SG1" in args["dataset"]:
    data_name = "STYLEGAN1"

elif "SG2" in args["dataset"]:
    data_name = "STYLEGAN2"

elif "celeba-hq_real_aligned" in args["dataset"]:
    data_name = "celeba-hq_real_aligned"

elif "FFHQ" in args["dataset"]:
    data_name = "ffhq_aligned"

elif "SG3" in args["dataset"]:
    data_name = "stylegan3-0.5_aligned"

elif "ADA" in args["dataset"]:
    data_name = "stylegan-ada-0.5_aligned"

else:
    data_name = "other"

# Which dataset to create?
cam_directory = "/scratch365/ccrum/Cyborg/augmented-cams_v3/" + data_name + "/" + args["network"] + "/" + model_training_name + "/" + args['gradcamType'] + "/" + args["size"] #+ "/"
if not os.path.exists(cam_directory):
    os.makedirs(cam_directory)

# Loading in the weights
weights = torch.load(args['modelPath'], map_location=device)

# Load weights of single binary DesNet121 model
#weights = torch.load(args['modelPath'], map_location=device)
if args['network'] == "resnet":
    im_size = 224
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
elif args['network'] == "inception":
    im_size = 299
    model = models.inception_v3(pretrained=True,aux_logits=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
elif args['network'] == "xception":
    im_size = 299
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
else: # else DenseNet
    im_size = 224
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)


#target_layers = [model.features[-1]]
# Loading the weights
model.load_state_dict(weights['state_dict'])
model = model.to(device)
model.eval()

# Network
if args['network'] == "xception":
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

imagesScores=[]
sigmoid = nn.Sigmoid()

directory = args["dataset"]

for root, dirs, files in os.walk(directory):
    for image in dirs:
        direct = root + "/" + image + "/*"
        print(direct)
        folder_name = image
        print(folder_name)
        for img_path in glob.glob(direct):
            print(img_path)
            try:
                #print(img_path)
                ############
                if args['network'] == "resnet":
                    im_size = 224
                    model = models.resnet50(pretrained=True)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, 2)
                elif args['network'] == "inception":
                    im_size = 299
                    model = models.inception_v3(pretrained=True,aux_logits=False)
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, 2)
                elif args['network'] == "xception":
                    im_size = 299
                    model, *_ = model_selection(modelname='xception', num_out_classes=2)
                else: # else DenseNet
                    im_size = 224
                    model = models.densenet121(pretrained=True)
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, 2)

                #target_layers = [model.features[-1]]
                # Loading the weights
                model.load_state_dict(weights['state_dict'])
                model = model.to(device)
                model.eval()
                ############
                #img_path = dirs + "/" + name
                #write_to_file("Theo.txt", img_path)

                # Naming conventions
                image_name = os.path.basename(img_path)

                # Loading in the image
                image = Image.open(img_path).convert('RGB')
                # Image transformation
                input_tensor = transform(image)
                image.close()

                # Generating the Grad-Cam
                if args['gradcamType'] == "gradcam":
                    #cam_extractor = GradCAM(model, target_layers)
                    cam_extractor = GradCAM(model)
                elif args['gradcamType'] == "gradcamPP":
                    cam_extractor = GradCAMpp(model, target_layers)

                out = model(input_tensor.unsqueeze(0))
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

                #plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
                #cam = resize(to_pil_image(activation_map[0].squeeze(0)).convert('L'), (224,224))
                cam = to_pil_image(activation_map[0].squeeze(0)).convert('L')
                if args['size'] == 'upscaled':
                    cam = cam.resize((im_size, im_size), resample = Image.BICUBIC)

                save_directory = cam_directory + "/" + str(folder_name)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                save_directory = save_directory + "/" + str(image_name)
                print(save_directory)
                #plt.imshow(cam)
                #plt.title("Blurred Image")
                #plt.show()

                cam.save(save_directory)
                data = "Cam saved at: " + str(save_directory)
                print(data)
            except:
                print("Error occured at " + str(image_name))
