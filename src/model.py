import torch
import torch.nn as nn
from utils import *
# from mlp import MLP
from matching import Matching

class Model(nn.Module):

    def __init__(self, batch_size, encoder):
        super(Model, self).__init__()
        self.encoder = encoder # ()
        n_classes = 3   # entailment, contradiction, neutral
        n_inputs = 4 * batch_size  # output dimension of Matching, input to MLP

        # classifier: Multi-Layer Perceptron with 1 hidden layer of 512 hidden units
        # dnn_hidden_units = [512]
        dnn_hidden_units = 512
        # classifier = MLP(n_inputs, dnn_hidden_units, n_classes)
        classifier = nn.Sequential(*[
            nn.Linear(n_inputs, dnn_hidden_units),
            nn.ReLU(),
            nn.Linear(dnn_hidden_units, n_classes),
        ])

        self.matching = Matching()

        self.net = nn.Sequential(*[
            # Matching(),
            classifier,
            torch.nn.Softmax(dim=1),
        ])
        # self.classifier = classifier
        # self.softmax = torch.nn.Softmax(dim=1)

    # def encode_embeddings(self, embeddings, lengths):
    #     # sort by decreasing length
    #     # sentence_dim = 1
    #     sorted_lengths, sorted_idxs = lengths.sort(0, descending=True)
    #     inverted_idxs = invert_idxs(sorted_idxs.numpy())
    #     sorted_input = embeddings[:, sorted_idxs]
    #     packed = nn.utils.rnn.pack_padded_sequence(sorted_input, sorted_lengths, batch_first=False)
    #     encoded = self.encoder(packed)
    #     (padded, _lengths) = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=False, padding_value=0.0, total_length=None)
    #     resorted = padded[inverted_idxs, :, :]
    #     return resorted

    def forward(self, premises, premise_lengths, hypotheses, hypothesis_lengths):
        # u = self.encode_embeddings(premises,      premise_lengths)
        # v = self.encode_embeddings(hypotheses, hypothesis_lengths)
        u = self.encoder(premises,      premise_lengths).mean(dim=0)
        v = self.encoder(hypotheses, hypothesis_lengths).mean(dim=0)
        matched = self.matching(u, v)
        return self.net(matched.transpose(0,1))
        # classified = self.classifier(matched.transpose(0,1))
        # return self.softmax(classified)
