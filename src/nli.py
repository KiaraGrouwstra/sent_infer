# import pickle
import argparse
import itertools
from functools import reduce
import numpy as np
# import torch
# import torch.nn
# import torchnlp
import torch.optim
from torchnlp.datasets import snli_dataset
from torchnlp.word_to_vector import GloVe
from nltk.tokenize import TreebankWordTokenizer
from tensorboardX import SummaryWriter
# from sent_eval import sent_eval, senteval_metrics  # batcher, prepare
from encoders import Baseline, lstms  # uni_lstm, bi_lstm
from model import Model
from timer_cm import Timer

def tokenize(snli_set):
    # TODO: does this add <s> </s> tags?
    tokenizer = TreebankWordTokenizer()
    token_set = set()
    for item in snli_set:
        for k in ['premise', 'hypothesis']:
            for token in tokenizer.tokenize(item[k]):
                token_set.add(token)
    return token_set

def get_embeddings(words):
    vectors = GloVe(cache='data/')
    glove_filtered = { k: vectors[k] for k in words if k in vectors }
    return glove_filtered

def get_data():
    snli_dir = 'data/snli_1.0/'
    sets = ['train', 'dev', 'test']
    return {k: [
        item for item in snli_dataset(snli_dir, **{k: True})[0:100] if item['label'] != '-'
    ] for k in sets}

def snli_glove(snli):
    # words = tokenize(test)  # all
    merge_sets = lambda x, y: {*x, *y}
    clean_text = lambda s: s.lower()
    token_sets = map(tokenize, snli.values())
    word_set = reduce(merge_sets, token_sets)
    words = set(map(clean_text, word_set))
    embeddings = get_embeddings(words)
    return embeddings

# snli = get_data()
# glove = snli_glove(snli)
