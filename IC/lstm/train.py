# script to train the caption generation decoder and evaluate the performance

#------------------------------------------------------------------------------

# imports
import argparse
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from build_vocab import Vocabulary
from data_loader import get_loader
from wat.caption import LSTMDecoder
from wat.utils import init_logger

#------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#------------------------------------------------------------------------------

def train_epoch(decoder, data_loader, criterion, optimizer, epoch, args):

    decoder.train()
    total_step = len(data_loader)
    epoch_loss = []

    for i, (features, captions, lengths) in enumerate(data_loader):

        features = features.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

        # Forward, backward and optimize
        decoder.zero_grad()
        optimizer.zero_grad()
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        # Print log info
        if i % args.log_step == 0:
            logger.info('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, args.num_epochs, i+1, total_step, loss.item(), np.exp(loss.item())))

        # Save the model checkpoints
        if (i + 1) % args.save_step == 0:
            logger.info(f'saving checkpoint to decoder-{epoch + 1}-{i + 1}.ckpt..')
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))

    mean_epoch_loss = np.mean(epoch_loss)

    return mean_epoch_loss

#------------------------------------------------------------------------------

def eval_epoch(decoder, data_loader, criterion, epoch, args):

    decoder.eval()
    total_step = len(data_loader)
    epoch_loss = []

    for i, (features, captions, lengths) in enumerate(data_loader):

        features = features.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

        # Forward only
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)

        epoch_loss.append(loss.item())

        # Print log info
        if i % args.log_step == 0:
            logger.info('Eval Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, args.num_epochs, i+1, total_step, loss.item(), np.exp(loss.item())))

    mean_epoch_loss = np.mean(epoch_loss)

    return mean_epoch_loss

#------------------------------------------------------------------------------


def main(args):

    init_logger(args.log_file_path)
    args_str = 'args:\n{\n'
    for k, v in args.__dict__.items():
        args_str += f'\t{k}: {v}\n'
    args_str += '}\n'
    logger.info(args_str)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Load vocabulary
    logger.info("loading vocabulary..")
    with open(args.vocab_path, 'rb') as f:
        vocab: Vocabulary = pickle.load(f)

    #------------------------------------------------------------------------------

    # Build data loaders
    train_image_dir = os.path.join(args.image_dir, 'train')
    train_caption_path = os.path.join(args.caption_path, 'hindi-visual-genome-train.txt')
    #MVG
    #train_caption_path = os.path.join(args.caption_path, 'malayalam-visual-genome-train.txt')

    train_loader = get_loader(train_image_dir, train_caption_path, vocab,\
        args.batch_size, shuffle=True, num_workers=args.num_workers)

    dev_image_dir = os.path.join(args.image_dir, 'dev')
    dev_caption_path = os.path.join(args.caption_path, 'hindi-visual-genome-dev.txt')
    #MVG
    #dev_caption_path = os.path.join(args.caption_path, 'malayalam-visual-genome-dev.txt')

    dev_loader = get_loader(dev_image_dir, dev_caption_path, vocab,\
        args.batch_size, shuffle=False, num_workers=args.num_workers)

    #------------------------------------------------------------------------------

    # Build the model
    decoder = LSTMDecoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    #------------------------------------------------------------------------------

    # Train and validate the model

    train_loss = []
    val_loss = []
    best_loss = np.inf

    for epoch in range(1, args.num_epochs+1):

        # train epoch    
        train_epoch_loss = train_epoch(decoder, train_loader, criterion, optimizer, epoch, args)
        train_loss.append(train_epoch_loss)

        # validate epoch
        val_epoch_loss = eval_epoch(decoder, dev_loader, criterion, epoch, args)
        val_loss.append(val_epoch_loss)

    logger.info(f'saving final checkpoint to decoder-final.ckpt..')
    torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-final-mvg.ckpt'))

    #np.savetxt(os.path.join(args.log_file_path, 'train_loss_epoch.txt'), np.array(train_loss), delimiter=',')
    np.savetxt('/idiap/temp/sparida/wat2021/log/train_loss_epoch.txt', np.array(train_loss), delimiter=',')
    #np.savetxt(os.path.join(args.log_file_path, 'val_loss_epoch.txt'), np.array(val_loss), delimiter=',')
    np.savetxt('/idiap/temp/sparida/wat2021/log/val_loss_epoch.txt', np.array(val_loss), delimiter=',')

#------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='expt/models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='expt/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        #default='/Users/subhadarshi/Downloads/wat-20210417T154726Z-001/wat/resnet50/l3',
                        #default='/idiap/temp/sparida/wat2021_bak/wat/resnet50/l3',
                        #VGG features
                        default='/idiap/temp/kkotwal/wat/vgg19/',
                        help='directory for image features')
    #parser.add_argument('--caption_path', type=str, default='data/hindi-visual-genome-11',
    parser.add_argument('--caption_path', type=str, default='/idiap/temp/sparida/wat2021_bak1_20Apr/data/hindi-visual-genome-11',
    #MVG
    #parser.add_argument('--caption_path', type=str, default='/idiap/temp/sparida/Data/mvg',
                        help='path for captions')
    parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=226, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--log_file_path', type=str, default=None,
    #parser.add_argument('--log_file_path', type=str, default='log',
                        help='path to log file. prints to console if set to None.')
    args = parser.parse_args()
    main(args)
