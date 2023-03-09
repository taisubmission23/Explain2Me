# SSIM.py

import os
import cv2
import pandas as pd
import glob
import argparse
import skimage.io
import skimage.filters
import torch
import itertools

def write_to_file(file_path, data):
    file = open(file_path, "a")
    file.write(data + "\n")
    file.close()

def findsubsets(s, n):
        return list(itertools.combinations(s, n))

# Driver Code
s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
n = 2

subset = findsubsets(s, n)

parser = argparse.ArgumentParser()
device = torch.device('cpu')

parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-m', '--model', default="xent")

args = vars(parser.parse_args())

if args["model"] == "cyborg":
        model = "/resnet/resnet_cyborg_"
        #model = "/densenet/densenet_cyborg_"

elif args["model"] == "xent":
        model = "/resnet/resnet_xent_"
        #model = "/densenet/densenet_xent_"

main_directory = "/home/ccrum/*"

dataset = args["dataset"]

prefix_path = "/scratch365/ccrum/Cyborg/test-cams/gradcam/"

files_dict = {}

outfile = "./results/" + dataset + "-" + args["model"] + ".csv"

for i in subset:
    first_model = prefix_path + dataset + model + str(i[0]) + "/small/*"
    second_model = prefix_path + dataset + model + str(i[1]) + "/small/"
    first_model_name = model + str(i[0])
    second_model_name = model + str(i[1])

    for cam_file in glob.glob(first_model):
        cam_1 = cv2.imread(cam_file, cv2.IMREAD_UNCHANGED)
        cam_name = os.path.basename(cam_file)
        cam_file_2 = second_model + cam_name
        cam_2 = cv2.imread(cam_file_2, cv2.IMREAD_UNCHANGED)
        try:
            ssim = skimage.metrics.structural_similarity(cam_1, cam_2)
        except:
            ssim = "NA"
        files_dict.setdefault(cam_name, []).append(ssim)
    pd.DataFrame.from_dict(data=files_dict, orient='index').to_csv(outfile, header=False)

    