import torch
import torch.nn as nn
from utils import *
from mlp import MLP
from matching import Matching

class Model(nn.Module):

    def __init__(self, n_inputs, encoder):
        super(Model, self).__init__()

        n_classes = 3   # entailment, contradiction, neutral

        self.premise_encoder = encoder()
        self.hypothesis_encoder = encoder()

        # classifier: Multi-Layer Perceptron with 1 hidden layer of 512 hidden units
        dnn_hidden_units = [512]
        classifier = MLP(n_inputs, dnn_hidden_units, n_classes)

        matching = Matching()

        self.net = nn.Sequential(*[
            Matching(),
            classifier,
            torch.nn.Softmax(dim=n_classes),
        ])

    def encode_embeddings(self, encoder, embeddings, lengths):
        # sort by decreasing length
        # sentence_dim = 1
        sorted_lengths, sorted_idxs = lengths.sort(0, descending=True)
        inverted_idxs = invert_idxs(sorted_idxs)
        sorted_input = embeddings[:, sorted_idxs]
        packed = nn.utils.rnn.pack_padded_sequence(sorted_input, sorted_lengths, batch_first=False)
        encoded = encoder(packed)
        (padded, _lengths) = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=False, padding_value=0.0, total_length=None)
        resorted = padded[inverted_idxs, :, :]
        return resorted

    def forward(self, premises, premise_lengths, hypotheses, hypothesis_lengths):
        u = self.encode_embeddings(self.   premise_encoder, premises,      premise_lengths)
        v = self.encode_embeddings(self.hypothesis_encoder, hypotheses, hypothesis_lengths)
        return self.net(u, v)
