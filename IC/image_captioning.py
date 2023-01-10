import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
from os import listdir
import csv

import torch
from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel, GPT2TokenizerFast

# load a fine-tuned image captioning model and corresponding tokenizer and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

PATH_DIR = '/content/drive/MyDrive/Sampled_Images/'
total_images = 11614
 
# get the path/directory
folder_dir = PATH_DIR

captioned_count = 0
duplicate_image_id = []
caption_dict = {}

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model = model.to(device)

for image in tqdm(os.listdir(folder_dir)):
  if image in caption_dict.keys():
    continue

  try:
    image_path = PATH_DIR + image
    im = Image.open(image_path) 
    im = im.resize((224,224) ,resample = Image.LANCZOS)

    pixel_values = feature_extractor(im, return_tensors="pt").pixel_values.to(device)

    # autoregressively generate caption (uses greedy decoding by default)
    #this uses beam_search=1
    generated_ids = model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  except:
    continue

  captioned_count += 1
  if image not in caption_dict.keys():
    caption_dict[image] = generated_text
  else:
    duplicate_image_id.append(image)

#write dict to csv file 
header = ['image_id', 'caption']

with open('image_captions.csv', 'a', encoding = 'UTF8') as fp:
  writer = csv.writer(fp)

  writer.writerow(header)

  for image in caption_dict.keys():
    row = [image, caption_dict[image]]
    writer.writerow(row)

