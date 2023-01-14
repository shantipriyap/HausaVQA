###################################################
# Build vocabulary using Hausa Visual Genome Dataset
##################################################
import nltk
import pickle
import argparse
from collections import Counter
from tqdm import tqdm
import numpy as np


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""

    words = []
    text = load_doc(json)
    text = text.split('\n')
    num_lines = len(text)
    #process_lines = 300000
    process_lines = 32923
    print('no. of lines:', num_lines)

    #for sentences in tqdm(text.split('\n'), desc='build vocabulary', unit=' sentences'):
    for i, sentences in enumerate(text[:process_lines]):

        if not (i%1000):
            print('processed', i, 'lines of', process_lines)

        splitted = sentences.split('\t')
        if len(splitted) < 4:
            hi_caption = splitted[0]
        elif len(splitted) == 7:
            hi_caption = splitted[6]

        tokens = nltk.tokenize.word_tokenize(hi_caption.lower())
        for item in tokens:
            words.append(item)

    wordList = np.unique(words).tolist()
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(wordList):
        #print("word:",word)
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='/home/jupyter/HausaVQA/IC/lstm/data/havg.ha',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='expt/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
