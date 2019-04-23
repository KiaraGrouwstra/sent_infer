import torch
import torch.nn as nn
from utils import *
from mlp import MLP
from matching import Matching
from pdb import set_trace

# constants
GLOVE_DIMS = 300  # glove embedding dimensions

class Model(nn.Module):

    def __init__(self, batch_size, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        n_classes = 3   # entailment, contradiction, neutral
        n_inputs = 4 * GLOVE_DIMS  # output dimension of Matching, input to MLP
        words_dim = 0

        # classifier: Multi-Layer Perceptron with 1 hidden layer of 512 hidden units
        # dnn_hidden_units = [512]
        # classifier = MLP(n_inputs, dnn_hidden_units, n_classes)
        dnn_hidden_units = 512
        classifier = nn.Sequential(*[
            nn.Linear(n_inputs, dnn_hidden_units),
            nn.ReLU(),
            nn.Linear(dnn_hidden_units, n_classes),
        ])

        self.matching = Matching()
        self.classifier = classifier
        self.softmax = torch.nn.Softmax(dim=words_dim)

        self.net = nn.Sequential(*[
            # Matching(),
            self.classifier,
            self.softmax,
        ])

    def forward(self, premises, premise_lengths, hypotheses, hypothesis_lengths):
        u = self.encoder(premises,      premise_lengths)
        v = self.encoder(hypotheses, hypothesis_lengths)
        matched = self.matching(u, v)
        return self.net(matched)
        # classified = self.classifier(matched)
        # return self.softmax(classified)
