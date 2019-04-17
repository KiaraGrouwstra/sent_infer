import torch
import torch.nn as nn

# 3 matching methods are applied to extract relations between u and v
class Matching(nn.Module):

    def forward(self, u, v):
        # concatenation of the two representations (u, v)
        concat = torch.cat([u, v], dim=0)
        # element-wise product u ∗ v
        product = torch.mul(u, v)
        # absolute element-wise difference |u − v|
        diff = torch.abs(u - v)
        features = [concat, product, diff]
        return torch.cat(features, dim=0)
