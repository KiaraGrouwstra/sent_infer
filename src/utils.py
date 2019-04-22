import itertools
import functools
import numpy as np
import pickle
import dill
from anycache import anycache
import argparse
import itertools
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torchtext.datasets import SNLI
from torchtext.vocab import GloVe
from torchtext.data import Field, BucketIterator  # , batch
from nltk.tokenize import TreebankWordTokenizer
from tensorboardX import SummaryWriter
# from utils import *  # accuracy, eval_dataset  # , oh_encode
# from dataset import DataSet
# from sent_eval import sent_eval, senteval_metrics  # batcher, prepare
# from encoders import Baseline, lstms  # uni_lstm, bi_lstm
# from model import Model
from operator import itemgetter
from timer_cm import Timer
from pdb import set_trace
from tqdm import tqdm
from joblib import Memory
from timeit import default_timer as timer

# https://docs.python.org/3/library/itertools.html
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))

# https://stackoverflow.com/a/30088323/1502035
def intersperse(e, l):    
    return list(itertools.chain(*[(i, e) for i in l]))[0:-1]

def invert_idxs(idxs):
    rng = list(range(len(idxs)))
    inv = dict(zip(idxs, rng))
    return [inv[i] for i in rng]

# accuracy function from deep learning practical
def accuracy(predictions, targets):
    return (predictions.argmax(dim=-1) == targets.argmax(dim=-1)).float().mean().detach().data.cpu().item()

# TODO: don't mutate
# mutates!
def pick_samples(ds, n):
    ds.examples = ds.examples[0:n]
    return ds

# # mutates!
# def filter_samples(ds, fn):
#     ds.examples = list(filter(fn, ds.examples))
#     return ds

def unpack_tokens(tpl):
    (words_batch, lengths) = tpl
    return (embeddings(words_batch), lengths)  # .to(torch.float)

def batch_cols(batch):
    prem_embeds, prem_lens = unpack_tokens(batch.premise)
    hyp_embeds,   hyp_lens = unpack_tokens(batch.hypothesis)
    labels = batch.label  # .to(torch.long)
    return (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels)

def prep_torch():
    # make it deterministic for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # pytorch defaults
    dtype = torch.cuda.FloatTensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    torch.set_default_tensor_type(dtype)
    return device
