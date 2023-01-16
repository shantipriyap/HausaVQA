"""Implementation of dataset interface of HindiVisualGenome for Image captioning."""

# Imports
import logging
import math
import os

import PIL
import nltk
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

logging.getLogger(__name__)

COLUMN_NAMES = ['image_idx', 'x', 'y', 'width', 'height', 'en', 'hi']
IMG_WIDTH, IMG_HEIGHT = 256, 256


# ------------------------------------------------------------------------------

class HindiVisualGenome(Dataset):
    """Dataset for Hindi Visual Genome."""

    def __init__(
            self,
            name="HindiVisualGenome",
            image_directory=None,
            text_file=None,
            group='train',
            **kwargs):

        self.name = name
        self.group = group
        self.image_directory = image_directory
        self.text_file = text_file

        # read text file
        self.df = pd.read_csv(self.text_file, sep='\t', encoding='utf-8', header=None)
        self.df.rename(columns={idx: name for idx, name in enumerate(COLUMN_NAMES)}, inplace=True)
        orig_data_size = len(self.df)
        sane_ids = set()
        for idx, row in tqdm(self.df.iterrows(), desc='filter sane samples', total=len(self.df)):
            is_sane_sample = True
            for name in COLUMN_NAMES[:-2]:
                is_sane_sample = is_sane_sample and not (row[name] != row[name]) and (
                        isinstance(row[name], int) or isinstance(row[name], float) or row[name].isnumeric())
            if is_sane_sample:
                sane_ids.add(idx)
        self.df = self.df.iloc[list(sane_ids)]
        self.df.reset_index(inplace=True)
        if len(self.df) != orig_data_size:
            logging.info(f'{len(self.df)} out of {orig_data_size} samples are sane and considered')
        else:
            logging.info(f'all samples are sane and considered')
        for name in COLUMN_NAMES[:-2]:
            self.df[name] = self.df[name].astype(int)
        for name in COLUMN_NAMES[-2:]:
            self.df[name] = self.df[name].astype(str)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

        # obtain resized coordinates of bounding box
        x_scaled_list, y_scaled_list, width_scaled_list, height_scaled_list = [], [], [], []
        for idx in tqdm(range(len(self.df)), desc='compute resized bounding box coordinates'):
            image = self.load_image(idx)
            orig_image_width, orig_image_height = image.size
            scale_width, scale_height = IMG_WIDTH / orig_image_width, IMG_HEIGHT / orig_image_height
            x_scaled, y_scaled, width_scaled, height_scaled = math.floor(scale_width * self.df['x'][idx]), \
                                                              math.floor(scale_height * self.df['y'][idx]), \
                                                              math.ceil(scale_width * self.df['width'][idx]), \
                                                              math.ceil(scale_height * self.df['height'][idx])
            x_scaled_list.append(x_scaled)
            y_scaled_list.append(y_scaled)
            width_scaled_list.append(width_scaled)
            height_scaled_list.append(height_scaled)

        self.df['x_scaled'] = x_scaled_list
        self.df['y_scaled'] = y_scaled_list
        self.df['width_scaled'] = width_scaled_list
        self.df['height_scaled'] = height_scaled_list

    def __getitem__(self, idx):
        image = self.load_image(idx, resize=True)
        # apply transforms (convert to tensor and normalize)
        image = self.transform(image)

        # return the scaled values x, y, w, h for the bounding box after resizing
        return {
            'image': image,
            'x': self.df['x_scaled'][idx],
            'y': self.df['y_scaled'][idx],
            'w': self.df['width_scaled'][idx],
            'h': self.df['height_scaled'][idx],
            'en': self.df['en'][idx],
            'hi': self.df['hi'][idx],
        }

    def __len__(self):
        return len(self.df)

    def load_image(self, idx, resize=False):
        image_idx = self.df['image_idx'][idx]
        image_path = os.path.join(self.image_directory, f'{image_idx}.jpg')
        # open image
        image = Image.open(image_path)  # PIL image
        if resize:
            # resize to IMG_WIDTH x IMG_HEIGHT
            image = image.resize((IMG_WIDTH, IMG_HEIGHT), PIL.Image.ANTIALIAS)
            # convert to RGB
            image = image.convert("RGB")
        return image

    def load_image_with_bounding_box(self, idx, resize=False):
        image = self.load_image(idx, resize=resize)
        draw = ImageDraw.Draw(image)
        if resize:
            x, y, width, height = self.df['x_scaled'][idx], self.df['y_scaled'][idx], self.df['width_scaled'][idx], \
                                  self.df['height_scaled'][idx]
        else:
            x, y, width, height = self.df['x'][idx], self.df['y'][idx], self.df['width'][idx], self.df['height'][idx]
        draw.rectangle(xy=(x, y, x + width, y + height), fill=None, outline='red', width=3)
        return image


class HindiVisualGenomeWithImageFeatures(Dataset):
    """Dataset for Hindi Visual Genome with image features."""

    def __init__(
            self,
            vocab,
            name="HindiVisualGenome",
            image_directory=None,
            text_file=None,
            group='train',  # unused
            **kwargs):

        self.vocab = vocab
        self.name = name
        self.group = group
        self.image_directory = image_directory
        self.text_file = text_file

        # read text file
        self.df = pd.read_csv(self.text_file, sep='\t', encoding='utf-8', header=None)
        self.df.rename(columns={idx: name for idx, name in enumerate(COLUMN_NAMES)}, inplace=True)
        orig_data_size = len(self.df)
        sane_ids = []
        self.image_ids = {}
        for idx, row in tqdm(self.df.iterrows(), desc='filter sane samples', total=len(self.df)):
            self.image_ids[idx] = row[COLUMN_NAMES[0]]
            is_sane_sample = True
            for name in COLUMN_NAMES[:-2]:
                is_sane_sample = is_sane_sample and not (row[name] != row[name]) and (
                        isinstance(row[name], int) or isinstance(row[name], float) or row[name].isnumeric())
            is_sane_sample = is_sane_sample and os.path.isfile(
                os.path.join(image_directory, f'{row[COLUMN_NAMES[0]]}.npy'))
            if is_sane_sample:
                sane_ids.append(idx)

        self.sane_ids = {k: v for k, v in enumerate(sane_ids)}
        if len(sane_ids) != orig_data_size:
            logging.info(f'{len(sane_ids)} out of {orig_data_size} samples are sane and considered')
        else:
            logging.info(f'all samples are sane and considered')

    def __getitem__(self, idx):

        # Convert caption (string) to word ids.
        caption = self.df['hi'][self.sane_ids[idx]]
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocab('<start>')]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))

        # load image features
        image_features = self.load_image_features(self.image_ids[self.sane_ids[idx]])

        return {
            'i_feat': torch.from_numpy(image_features['i_feat']),
            's_feat': torch.from_numpy(image_features['s_feat']),
            'caption': torch.LongTensor(caption),
        }

    def __len__(self):
        return len(self.sane_ids)

    def load_image_features(self, idx):
        image_features_path = os.path.join(self.image_directory, f'{idx}.npy')
        features = np.load(image_features_path, allow_pickle=True).item()
        return features
