import torch
import torch.nn as nn

# - glove baseline
class Baseline(nn.Module):
    '''average word embeddings to obtain sentence representations'''

    def __init__(self, words_length, words_dim):
        # words_length: number of words including padding
        super(Baseline, self).__init__()
        self.words_length = words_length
        self.words_dim = words_dim

    def forward(self, packed_seq):
        # words_embeddings: dimensions include embedding dimension, word, and sentence
        words_embeddings = packed_seq.data
        # words_used: number of words per sentence (excl. padding)
        words_used = packed_seq.batch_sizes
        return self.words_length / words_used * words_embeddings.mean(dim=self.words_dim)

def lstms(words_length):
    # LSTMs: https://github.com/VictorZuanazzi/Inference_Bot/blob/master/code/encoder.py
    lstm_conf = {
        'input_size': words_length,
        'hidden_size': 512,
        'num_layers': 1,
        'dropout': 0,
    }
    # - uni-directional LSTM
    # - bi-directional LSTM
    [uni_lstm, bi_lstm] = [nn.LSTM(**lstm_conf, bidirectional=bidir) for bidir in [False, True]]
    # - bi-directional LSTM + max pooling
    kernel_size = words_length
    max_bi_lstm = nn.Sequential(*[
        bi_lstm,
        nn.MaxPool1d(kernel_size),  # torch.max()?
    ])
    return [lambda: uni_lstm, lambda: bi_lstm, lambda: max_bi_lstm]
