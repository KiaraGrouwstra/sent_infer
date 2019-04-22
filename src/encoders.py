import torch
import torch.nn as nn
from utils import *
from pdb import set_trace

# - glove baseline
class Baseline(nn.Module):
    '''average word embeddings to obtain sentence representations'''

    def __init__(self, words_dim):  # , words_length
        # words_length: number of words including padding
        super(Baseline, self).__init__()
        # self.words_length = words_length
        self.words_dim = words_dim

    # def forward(self, packed_seq):
    def forward(self, embeddings, lengths):
        # words_embeddings: dimensions include embedding dimension, word, and sentence
        # words_embeddings = packed_seq.data
        # # words_used: number of words per sentence (excl. padding)
        # words_used = packed_seq.batch_sizes
        return words_length / words_used * embeddings.mean(dim=self.words_dim)  # TODO: fix
        # return words_embeddings.mean(dim=self.words_dim)

def lstms(input_size):
    # LSTMs: https://github.com/VictorZuanazzi/Inference_Bot/blob/master/code/encoder.py
    lstm_conf = {
        'input_size': input_size,
        'hidden_size': 512,
        'num_layers': 1,
        'dropout': 0,
    }
    # - uni-directional LSTM
    # - bi-directional LSTM
    [uni_lstm, bi_lstm] = [nn.LSTM(**lstm_conf, bidirectional=bidir) for bidir in [False, True]]
    # - bi-directional LSTM + max pooling
    kernel_size = input_size
    max_bi_lstm = nn.Sequential(*[
        bi_lstm,
        nn.MaxPool1d(kernel_size),  # torch.max()?
    ])
    # set_trace()
    # return [(lambda: LstmEncoder(lstm)) for lstm in [uni_lstm, bi_lstm, max_bi_lstm]]
    return [LstmEncoder(lstm) for lstm in [uni_lstm, bi_lstm, max_bi_lstm]]
    # uni = lambda: LstmEncoder(uni_lstm)
    # bi = lambda: LstmEncoder(bi_lstm)
    # mx = lambda: LstmEncoder(max_bi_lstm)
    # return (uni, bi, mx)

class LstmEncoder(nn.Module):

    def __init__(self, lstm):
        super(LstmEncoder, self).__init__()
        self.lstm = lstm

    def forward(self, embeddings, lengths):
        # sort by decreasing length
        # sentence_dim = 1
        sorted_lengths, sorted_idxs = lengths.sort(0, descending=True)
        inverted_idxs = invert_idxs(sorted_idxs.numpy())
        sorted_input = embeddings[:, sorted_idxs]
        packed = nn.utils.rnn.pack_padded_sequence(sorted_input, sorted_lengths, batch_first=False)
        (encoded, _) = self.lstm(packed)
        (padded, _lengths) = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=False, padding_value=0.0, total_length=None)
        # resorted = padded[inverted_idxs, :, :]
        resorted = padded[:, inverted_idxs, :]
        return resorted
