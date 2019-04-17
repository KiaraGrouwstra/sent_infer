import torch
import torch.nn as nn
# from utils import *
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

    def forward(self, item):
        u = self.   premise_encoder(item['premise'])
        v = self.hypothesis_encoder(item['hypothesis'])
        return self.net(u, v)
