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

def get_encoder(enc_type):
    (n_dim, words_dim, embedding_dim) = range(3)
    # words_length = words_embeddings.size[words_dim]
    input_size = GLOVE_DIMS  # what of number of words?
    # set_trace()
    [uni_lstm, bi_lstm, max_lstm] = lstms(input_size)
    if enc_type == 'lstm':
        encoder = uni_lstm
    elif enc_type == 'bilstm':
        encoder = bi_lstm
    elif enc_type == 'maxlstm':
        encoder = max_lstm
    else:  # baseline
        encoder = lambda: Baseline(words_dim)  # words_length
    return encoder

def eval_dataset(model, dataset, batch_size):
    cols = ['loss', 'acc']
    df = pd.DataFrame([], columns=cols)
    (iterator,) = BucketIterator.splits(datasets=(dataset,), batch_sizes=[batch_size], device=device, shuffle=True)
    for batch in tqdm(iterator):
        (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels) = batch_cols(batch)
        predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
        df = df.append([dict(zip(cols, [
            loss_fn( predictions, labels),
            accuracy(predictions, labels),
        ]))])
    (loss, acc) = list(df.mean())
    return (loss, acc)

def get_optimizer(optim, optim_pars):
    if optim == 'adadelta':
        optimizer = torch.optim.Adadelta(**optim_pars)
    elif optim == 'adagrad':
        optimizer = torch.optim.Adagrad(**optim_pars)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(**optim_pars)
    elif optim == 'adam':
        optimizer = torch.optim.Adam(**optim_pars)
    else:  # SGD
        optimizer = torch.optim.SGD(**optim_pars)
    return optimizer

# def optimize():
#     # TODO: investigate trixi: https://github.com/MIC-DKFZ/trixi/blob/master/examples/pytorch_experiment.ipynb
#     emins = lambda n: list(reversed([10**-(i+1) for i in range(n)]))

#     options = {
#         'dnn_hidden_units': ['10', '100', '200,200', '400', '25,50,50,50,50,25'],
#         'learning_rate': emins(5),
#         'max_steps': [500, 1000, 1500, 2500],
#         'batch_size': [25, 50, 100, 250],
#         'optimizer': ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta'],
#         'weight_decay': [0, *emins(5)],
#     }

#     def eval_loss(dct):
#         FLAGS = argparse.Namespace(**dct)
#         # ^ does this work to be in scope for train_mlp_pytorch?
#         print_flags()
#         return train()

#     run_ea = get_ea(options, eval_loss)
#     best_member, best_loss = run_ea(n=8, rounds=20)

def get_model(enc_type):
    encoder = get_encoder(enc_type)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = Model(GLOVE_DIMS, encoder)
    device = prep_torch()
    model.to(device)
    return model
