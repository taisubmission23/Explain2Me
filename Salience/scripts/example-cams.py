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

# Argparse
parser = argparse.ArgumentParser()

device = torch.device('cpu')
parser.add_argument('-imgf','--imageFolder', default="/scratch365/ccrum/Cyborg/Data/test/0_real/ffhq_aligned", type=str)
parser.add_argument('-mp', '--modelPath', default="/scratch365/ccrum/Cyborg/Cyborg_weights/densenet/densenet_xent_1/Logs/final_model.pth", type=str)
parser.add_argument('-n','--network', default="densenet", type=str)
parser.add_argument('-gt', '--gradcamType', default="gradcam", type=str)
parser.add_argument('-s', '--size', default="upscaled", type=str)



#args = parser.parse_args([])

args = vars(parser.parse_args())

#print(args)
# Creation of Cams
#path_sep = "/scratch365/ccrum/Cyborg/Cyborg_weights/"
path_sep = "/afs/crc.nd.edu/user/c/ccrum/Private/Research/CyborgGeneratedSaliency/Cyborg-mini/CYBORG_NGEBM-master/model_output_local/"
stripped = args['modelPath'].split(path_sep, 1)[0]
end_sep = "/Logs/final_model.pth"
folder = stripped.split(end_sep, 1)[0]

#print(folder)

if "STYLEGAN1" in args["imageFolder"]:
    data_name = "STYLEGAN1"

if "STYLEGAN2" in args["imageFolder"]:
    data_name = "STYLEGAN2"

if "celeba-hq_real_aligned" in args["imageFolder"]:
    data_name = "celeba-hq_real_aligned"

if "ffhq_aligned" in args["imageFolder"]:
    data_name = "ffhq_aligned"

if "stargan_aligned" in args["imageFolder"]:
    data_name = "stargan_aligned"

if "progan_aligned" in args["imageFolder"]:
    data_name = "progan_aligned"

if "stylegan3-0.5_aligned" in args["imageFolder"]:
    data_name = "stylegan3-0.5_aligned"

if "stylegan-ada-0.5_aligned" in args["imageFolder"]:
    data_name = "stylegan-ada-0.5_aligned"

if "original_data" in args["imageFolder"]:
    data_name = "original_data"

if "object" in args["imageFolder"]:
    degrad_type = "object"

if "background" in args["imageFolder"]:
    degrad_type = "background"

else:
    data_name = "other-half-training"

# Which dataset to create?
cam_directory = "/scratch365/ccrum/Cyborg/proxy-otherhalf-training-cams/" + args['gradcamType'] + "/"  + data_name + "/" + folder + "/" + args["size"] + "/" #+ degrad_type
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

directory = args['imageFolder']

directory = str(directory) + "/*"
print(directory)
# Starting the loop
#for img_path in resample(glob.glob(directory), n_samples=10000, replace=True, random_state=1):
for img_path in glob.glob(directory):
    try:
        img_path = "/scratch365/ccrum/Cyborg/Data/test/0_real/ffhq_aligned/00047.png"
        #for img_path in glob.glob(directory):
        ##########
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



        #########


        # Image path
        #print(image)
        image_name = os.path.basename(img_path)
        #img_path = directory[:-2] + "/" + image
        #print(img_path)

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
        #cam_extractor = SmoothGradCAMpp(model)
        #from torchcam.methods import CAM
        #cam_extractor = CAM(model)

        # Get your input
        #input_tensor = read_image(transformImage)
        # Preprocess it for your chosen model
        #input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        #input_tensor = transform(img)
        # Preprocess your data and feed it to the model
        out = model(input_tensor.unsqueeze(0))
        # Retrieve the CAM by passing the class index and the model output
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

        #plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
        #cam = resize(to_pil_image(activation_map[0].squeeze(0)).convert('L'), (224,224))
        cam = to_pil_image(activation_map[0].squeeze(0)).convert('RGB')
        if args['size'] == 'upscaled':
            cam = cam.resize((im_size, im_size), resample = Image.BICUBIC)

        #save_directory = args['network'] + "-xent-" + str(image_name)

        cam.save("densenet-xent-example.png")
        #cam.save("Test.png")
        print("Cam saved at: " + str(save_directory))
    except:
        print("Cam was skipped due to some error.")


