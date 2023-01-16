#############################################################
# Dataloader module, load data in batchwise manner
# Author: Shantipriya Parida (Silo AI)
#
#############################################################

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from build_vocab import Vocabulary
from dataset.hausa_vqa import HausaVQAWithImageFeatures


def collate_fn(batch):
    """Creates mini-batch tensors"""
    #this part of code changed to take part for whole image features with full sub image features
    #features = [torch.cat((0.5*sample['i_feat'], sample['s_feat']), dim=0) for sample in batch] 
    #features = [torch.cat((sample['i_feat'], sample['s_feat']), dim=0) for sample in batch]
    features = [sample['i_feat'] for sample in batch]
    features = torch.stack(features)
    captions = [sample['caption'] for sample in batch]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    lengths = [len(cap) for cap in [sample['caption'] for sample in batch]]
    lengths = torch.LongTensor(lengths)
    return features, captions, lengths


def get_loader(image_dir, caption_path, vocab: Vocabulary, batch_size, shuffle, num_workers):
    """Returns data loader"""

    dataset = HausaVQAWithImageFeatures(
        vocab=vocab,
        image_directory=image_dir,
        text_file=caption_path,
        group='dummy',
    )

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
