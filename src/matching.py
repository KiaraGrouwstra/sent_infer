import torch
import torch.nn as nn
from pdb import set_trace

# 3 matching methods are applied to extract relations between u and v
class Matching(nn.Module):

    # def __init__(self, dim):
    #     super(Matching, self).__init__()
    #     self.dim = dim

    def forward(self, u, v):
        # concatenation of the two representations (u, v)
        # TODO: decide dimension to concat on
        # TODO: wtf is up with 512 instead of 300?
        dim = 0
        # set_trace()
        concat = torch.cat([u, v], dim=dim)
        # element-wise product u ∗ v
        product = torch.mul(u, v)
        # absolute element-wise difference |u − v|
        diff = torch.abs(u - v)
        features = [concat, product, diff]
        return torch.cat(features, dim=dim)
