import os
import os
import glob
import argparse
import skimage.io
import skimage.filters
import matplotlib.image as mpimg
import skimage
import skimage.measure
import glob

# generic function to write results to file
def write_to_file(file_path, data):
    file = open(file_path, "a")
    file.write(data + "\n")
    file.close()

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-d','--dataset', help='Description for dataset argument', required=True)
parser.add_argument('-t', '--type', default="gradcam")

args = vars(parser.parse_args())

# type of salience
cam_type = args["type"]

# Output file
output_file = str(args['dataset'])
output_file = output_file.replace('/', '-')
output_file = output_file[1:]
output_file = "./test/" + output_file + ".txt"

# Specifying directory
directory = str(args['dataset'])
directory = directory + "/*"

# Dataset
dataset_name = args["dataset"]
dataset_name = dataset_name.split("/")[6]

# Training type
model_training_name = args["dataset"]
model_training_name = model_training_name.split("/")[8]

for cam in glob.glob(directory):
    cam_path = os.path.join(directory, cam)
    cam_name = os.path.basename(cam_path)
    cam = mpimg.imread(cam_path)
    entrophy = skimage.measure.shannon_entropy(cam)            
    data = str(cam_name)  + "," + str(entrophy) + "," + str(model_training_name) + "," + str(dataset_name)  + "," + str(cam_type)
    print(data)
    write_to_file(output_file, data)
