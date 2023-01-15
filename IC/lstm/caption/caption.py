"""The Caption module is transformer or LSTM (encoder-decoder) module. It
accepts the inputs-- (1) N different regions (in the form of class/label
information obtained from the RCNN; and (2) matrix or graph (NxN) or attention
module that defines the (q,k,v) tuples. Note that in our case, only one `q`
(query) roi exists. The `k, v` values are obtained from previous module. """

# imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ------------------------------------------------------------------------------

# LSTM Decoder to generate caption
# The below code is referred from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

class LSTMDecoder(nn.Module):
    """LSTM based caption generation model"""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
        #self.img_fc = nn.Linear(1024, embed_size) # for L3: only s_feat
        #self.img_fc = nn.Linear(2048 + 1024, embed_size) # for L3: s + i feat
        #self.img_fc = nn.Linear(2048 + 1024, embed_size) # for L3: s + i feat
        self.img_fc = nn.Linear(2048 + 2048, embed_size) # for VGG19

        #self.img_fc = nn.Linear(2048, embed_size)  # for L4: only s_feat
        #self.img_fc = nn.Linear(2048 + 2048, embed_size) # for L4: s + i feat


    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        features = self.img_fc(features)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        features = self.img_fc(features)
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
