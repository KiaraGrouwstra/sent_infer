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
from torchtext.data import Field, BucketIterator, Dataset  # , batch
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

def pick_samples(ds, n):
    examples = ds.examples[0:n]
    return Dataset(examples, ds.fields)

def unpack_tokens(tpl, text_embeds):
    (words_batch, lengths) = tpl
    return (text_embeds(words_batch), lengths)

def batch_cols(batch, text_embeds):
    prem_embeds, prem_lens = unpack_tokens(batch.premise   , text_embeds)
    hyp_embeds,   hyp_lens = unpack_tokens(batch.hypothesis, text_embeds)
    labels = batch.label
    return (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels)

# def embed_tokens(tokens, stoi, text_embeds, size, pad_idx):
#     idxs = list(map(lambda token: stoi[token], tokens)) + [pad_idx] * (size - len(tokens))
#     return text_embeds(torch.LongTensor(idxs))

# def batch_tokens(sentences, text_field, text_embeds):
#     stoi = text_field.vocab.stoi
#     pad_idx = stoi[text_field.pad_token]
#     lens = list(map(len, sentences))
#     size = max(lens)
#     embeds = torch.stack([embed_tokens(x, stoi, text_embeds, size, pad_idx) for x in sentences])
#     lens_ = torch.LongTensor(lens)
#     return (embeds, lens_)

# def batch_rows(batch, text_field, label_vocab, text_embeds):
#     labels = torch.LongTensor([label_vocab.stoi[x.label] for x in batch])
#     (prem_embeds, prem_lens) = batch_tokens([x.premise    for x in batch], text_field, text_embeds)
#     (hypo_embeds, hypo_lens) = batch_tokens([x.hypothesis for x in batch], text_field, text_embeds)
#     return (prem_embeds, prem_lens_, hypo_embeds, hypo_lens_, labels)

def prep_torch():
    # make it deterministic for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # pytorch defaults
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    return device
