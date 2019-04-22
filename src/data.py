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
from utils import *  # accuracy, eval_dataset  # , oh_encode
# from dataset import DataSet
# from sent_eval import sent_eval, senteval_metrics  # batcher, prepare
from encoders import Baseline, lstms  # uni_lstm, bi_lstm
from model import Model
from operator import itemgetter
from timer_cm import Timer
from pdb import set_trace
from tqdm import tqdm
from joblib import Memory
from timeit import default_timer as timer

# constants
GLOVE_DIMS = 300  # glove embedding dimensions
MEMORY = Memory(location='cache/', verbose=0)


# @MEMORY.cache  # 30s...
@anycache(cachedir='/tmp/glove.cache')  # 10s :)
def get_glove():
    return GloVe(dim=GLOVE_DIMS)

# super fast, probably doesn't even need caching!
# @anycache(cachedir='/tmp/embeds.cache')
@MEMORY.cache
def get_embeds(vectors):
    embeddings = nn.Embedding.from_pretrained(vectors)
    embeddings.requires_grad = False
    return embeddings

# def get_snli():
#     return SNLI.splits(text_field, label_field)

# @MEMORY.cache  # errors
def get_snli(text_field, label_field):
    # also filters out unknown label '-'! :)
    return SNLI.splits(text_field, label_field)

# @anycache(cachedir='/tmp/both.cache')  # errors?
# @MEMORY.cache  # doesn't actually seem to make it faster, 110s-ish
def get_data():
    '''returns: (train, dev, test)'''
    with Timer('glove') as timer:
        print('glove{')
        # glove = GloVe(dim=GLOVE_DIMS)
        glove = get_glove()
        print('}')
    tokenizer = TreebankWordTokenizer().tokenize
    text_field = Field(sequential=True, tokenize=tokenizer, include_lengths=True, lower=True, use_vocab=True)
    label_field = Field(sequential=False, pad_token=None, unk_token=None, is_target=True, use_vocab=True)
    with Timer('snli') as timer:
        print('snli{')

        splits = get_snli(text_field, label_field)

        # fn = 'snli.obj'
        # try:
        #     with open(fn, 'rb') as fp:
        #         # splits = pickle.load(fp)
        #         with timer.child('load'):
        #             splits = dill.load(fp)
        # except EOFError:  # FileNotFoundError
        #     with timer.child('define'):
        #         # splits = SNLI.splits(text_field, label_field)
        #         splits = get_snli(text_field, label_field)
        #         # splits = get_snli()
        #     with timer.child('dump'):
        #         with open(fn, 'wb') as fp:
        #             # pickle.dump(splits, fp)
        #             dill.dump(splits, fp)

        print('}')

    # (train, dev, test) = splits
    text_field.build_vocab(*splits, vectors=glove)
    label_field.build_vocab(*splits)
    text_vocab = text_field.vocab
    label_vocab = label_field.vocab

    with Timer('embeddings') as timer:
        print('embeddings{')
        # embeddings = nn.Embedding.from_pretrained(text_field.vocab.vectors)
        # embeddings.requires_grad = False
        text_embeds  = get_embeds(text_vocab.vectors)
        # label_embeds = get_embeds(label_vocab.vectors)
        print('}')
    snli = [pick_samples(ds, n=100) for ds in splits]  # TODO: comment

    return (snli, text_field, label_vocab, text_embeds)
    # , label_embeds
