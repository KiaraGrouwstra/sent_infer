import torch
import torch.nn as nn
from utils import *
from pdb import set_trace

# constants
GLOVE_DIMS = 300  # glove embedding dimensions

# - glove baseline
class Baseline(nn.Module):
    '''average word embeddings to obtain sentence representations'''

    def __init__(self):
        super(Baseline, self).__init__()

    def forward(self, embeddings, lengths):
        words_dim = 0
        return embeddings.sum(dim=words_dim) / lengths.view(-1, 1).float()

    def get_dim(self):
        return GLOVE_DIMS  # output dimension of Matching, input to MLP

def lstms(input_size):
    # LSTMs: https://github.com/VictorZuanazzi/Inference_Bot/blob/master/code/encoder.py
    hidden_size = 512
    lstm_conf = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': 1,
        'dropout': 0,
    }
    # - uni-directional LSTM
    # - bi-directional LSTM
    [uni_lstm, bi_lstm] = [nn.LSTM(**lstm_conf, bidirectional=bidir) for bidir in [False, True]]
    # - bi-directional LSTM + max pooling
    kernel_size = input_size
    max_bi_lstm = bi_lstm
    # max_bi_lstm = nn.Sequential(*[
    #     bi_lstm,
    #     nn.MaxPool1d(kernel_size),  # torch.max()?
    # ])
    return [LstmEncoder(lstm) for lstm in [uni_lstm, bi_lstm, max_bi_lstm]]

class LstmEncoder(nn.Module):

    def __init__(self, lstm):
        super(LstmEncoder, self).__init__()
        self.lstm = lstm
        multiplier = 2 if lstm.bidirectional else 1
        self.dim = self.lstm.hidden_size * multiplier

    def forward(self, embeddings, lengths):
        # sort by decreasing length
        sorted_lengths, sorted_idxs = lengths.sort(0, descending=True)
        inverted_idxs = invert_idxs(sorted_idxs.numpy())
        sorted_input = embeddings[:, sorted_idxs]
        packed = nn.utils.rnn.pack_padded_sequence(sorted_input, sorted_lengths, batch_first=False)
        (encoded, _) = self.lstm(packed)
        (padded, _lengths) = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=False, padding_value=0.0, total_length=None)
        resorted = padded[:, inverted_idxs, :]
        return resorted.mean(dim=0)

    def get_dim(self):
        return self.dim
