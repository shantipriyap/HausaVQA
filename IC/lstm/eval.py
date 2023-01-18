############################################################################
# This is the evaluation file for the HVG caption generation
# Input the Evaluation and Challenge Test Set for output caption
# Author: Shantipriya Parida (Silo AI, Finland)
############################################################################
import argparse
import logging
import os
import pickle

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from build_vocab import Vocabulary
from data_loader import get_loader
from caption import LSTMDecoder
from utils import init_logger

logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    init_logger(args.log_file_path)
    args_str = 'args:\n{\n'
    for k, v in args.__dict__.items():
        args_str += f'\t{k}: {v}\n'
    args_str += '}\n'
    logger.info(args_str)

    # Load vocabulary
    logger.info("loading vocabulary..")
    with open(args.vocab_path, 'rb') as f:
        vocab: Vocabulary = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Build the model
    decoder = LSTMDecoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    decoder.load_state_dict(torch.load(args.model_path, map_location=device))

    # Define criterion
    criterion = torch.nn.CrossEntropyLoss()

    av_loss = 0.0
    generated_captions = []
    with torch.no_grad():
        for i, (features, captions, lengths) in enumerate(tqdm(
                data_loader, desc='generate captions', unit=' batches',
        )):
            features = features.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

            # Generate caption
            sampled_ids = decoder.sample(features)  # (batch_size, max_seq_length)

            for generated_ids in sampled_ids.detach().cpu().tolist():
                # Convert word_ids to words
                generated_caption = []
                for word_id in generated_ids[1:]:  # skip <start> token
                    word = vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    generated_caption.append(word)
                generated_caption = ' '.join(generated_caption)
                generated_captions.append(generated_caption)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            av_loss += loss.item()
    #logger.info(f'Average loss: {av_loss / (i + 1):.4f}')
    logger.info(f'Writing generated captions to {args.output_path}..')
    if args.output_path != '':
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.writelines(list(map(lambda s: s + '\n', generated_captions)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='expt/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--model_path', type=str, default='expt/models/decoder-final-ha.ckpt',
                        help='path to trained model')
    parser.add_argument('--image_dir', type=str,
                        default='/home/jupyter/HausaVQA/IC/lstm/data/resnet50/', help='')
    parser.add_argument('--caption_path', type=str,
                        default='/home/jupyter/HausaVQA/IC/lstm/data/hausa-test1.txt',
                        help='path to captions file')
    parser.add_argument('--output_path', type=str, default='expt/output/test_out.txt', help='output file path')

    # Model parameters (should be same as parameters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in lstm')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--log_file_path', type=str, default=None,
                        help='path to log file. prints to console if set to None.')
    args = parser.parse_args()
    main(args)
