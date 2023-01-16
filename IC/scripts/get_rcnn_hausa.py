# a script to run RCNN on list of images defined by a folder and text file.

# imports
import os
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image

from torchvision.models import resnet50
from torchvision import transforms


IMAGE_DIR = "<provide image directory here>"
FILE_LIST = "<provide xls or csv file containing image ids>"
OUTPUT_DIRECTORY = "<provide location to save features>"
IMAGE_SIZE = 224

#------------------------------------------------------------------------------
def get_unique_images(filepath):

    fs = pd.read_excel(filepath)
    print(f"# total entries in dataset: {len(fs)}")
    
    unique_images = []   
    for i in range(len(fs)):
         if type(fs.loc[i].ans_ha) == str or type(fs.loc[i].ques_ha) == str:
             unique_images.append(str(fs.loc[i].image_id))        
    print(f"# total validated entries: {len(unique_images)}")             
    
    unique_images = list(set(unique_images))
    print(f"# unique images: {len(unique_images)}")
    
    return sorted(unique_images)
    
#-------------------------------


class RCNNExtractor(object):

    def __init__(self, output_directory):

        self.network = resnet50(pretrained=True)
        self.network.eval()

        self.i_feat_layer = "avgpool"

        self.output_directory = output_directory

        self.data_transform = transforms.Compose([\
            transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),\
            transforms.ToTensor(),\
            transforms.Normalize(mean=[0.485, 0.456, 0.406],\
            std=[0.229, 0.224, 0.225]), ])

#------------------------------------------------------------------------------

    def preprocess(self, image):

        if image.mode != "RGB":
            print("Image is grayscale. Converting to RGB")
            image = image.convert(mode='RGB')

        image_tensor = self.data_transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

#-------------------------------------------------------------------------------

    def process_image(self, x):

        # obtain image features
        for name, layer in self.network._modules.items():

            if name == "fc":
                break

            x = layer(x)

            if name == self.i_feat_layer:
                i_feat = x

        # convert to numpy array
        s_feat = np.array([0]) # placeholder for region features (not needed)
        i_feat = i_feat.detach().numpy().flatten()

        #print("samples features:", i_feat[:10])

        return s_feat, i_feat

#------------------------------------------------------------------------------

    def process(self, image_directory):

        lines = get_unique_images(FILE_LIST)
        total = len(lines)
        print(total)
        
        for idx, line in enumerate(lines[:]):

            temp_l = line          
            output_file = os.path.join(self.output_directory, temp_l + '.npy')

            if os.path.exists(output_file):
                pass
                print("  --> File exists. Skipping.")

            else:
            
                features = None

                image_path = os.path.join(image_directory, temp_l + '.jpg')
                image = Image.open(image_path)
                orig_shape = image.size

                image_tensor = self.preprocess(image)

                try:           
                    features = self.process_image(image_tensor)
                except Exception as e:
                    print("Exception: ", e)

                if features is not None:
                    out_dict = {'s_feat': features[0], 'i_feat': features[1]}

                    with open(output_file, 'wb') as f:
                        np.save(f, out_dict)

        return


#------------------------------------------------------------------------------


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    img_ext = RCNNExtractor(OUTPUT_DIRECTORY)

    img_ext.process(IMAGE_DIR)

#------------------------------------------------------------------------------

