import os
import pickle
# import dill
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
from model_utils import *
from data import *

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = 'mean', help='model, baseline (default), lstm, bilstm, maxlstm')
    parser.add_argument('--model_name', type = str, default = 'bilstm_optim=adam_dim=2048', help='model name, default bilstm_optim=adam_dim=2048')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoint_folder/', help='check point_path, default checkpoint_folder')
    parser.add_argument('--train_data_path', type = str, default = 'snli/train.tsv', help='train, default snli/train.tsv')

    parser.add_argument('--learning_rate', type = float, default = 0.1,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type = int, default = 500,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type = int, default = 25,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--weight_decay', type = float, default = 1e-3,
                        help='weight decay used in the optimizer, default 1e-3')
    parser.add_argument('--learning_decay', type = int, default = 5,
                        help='by what to divide the LR when accuracy improves, default 5')
    parser.add_argument('--learning_threshold', type = int, default = 1e-5,
                        help='at which learning rate to stop the experiment, default 1e-5')
    parser.add_argument('--optimizer_type', type = str, default = 'SGD',
                        help='optimizer, default SGD, also supports adam, adagrad, rmsprop, adadelta')

    flags, unparsed = parser.parse_known_args()
    return flags

def train():
    # constants
    flags = parse_flags()
    flag_keys = ['model_type', 'model_name', 'checkpoint_path', 'train_data_path', 'learning_rate', 'max_epochs', 'batch_size', 'eval_freq', 'weight_decay', 'learning_decay', 'learning_threshold', 'optimizer_type']
    (model_type, model_name, checkpoint_path, train_data_path, lr, max_epochs, batch_size, eval_freq, weight_decay, learning_decay, learning_threshold, optim) = itemgetter(*flag_keys)(vars(flags))

    loss_fn = torch.nn.CrossEntropyLoss()
    (model, optimizer) = make_model(model_type, lr, weight_decay, optim, batch_size)
    # iterate
    prev_acc = 0.0
    device = prep_torch()
    (snli, text_field, label_vocab, text_embeds) = get_data()
    (train, dev, test) = snli
    # with SummaryWriter(model_name) as w:
    for epoch in tqdm(range(max_epochs)):
        optimizer.zero_grad()

        # train
        (train_iter,) = BucketIterator.splits(datasets=(train,), batch_sizes=[batch_size], device=device, shuffle=True)
        # train_iter.create_batches()
        # batch = next(train_iter.batches)
        for batch in tqdm(train_iter):
        # for epoch in range(max_epochs):
            # with open('batch_ex.pkl', 'rb') as f:
            #     batch = pickle.load(f)
            (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels) = batch_cols(batch, text_embeds)
            # (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels) = batch_rows(batch, text_field, label_vocab, text_embeds)
            # set_trace()
            predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
            train_loss = loss_fn(predictions, labels)
            train_acc = accuracy(predictions, labels)

            # evaluate on dev set and report results
            if epoch % eval_freq == 0:
                (dev_loss, dev_acc) = eval_dataset(model, dev, batch_size, loss_fn)
                print(dev_acc)

            # # training is stopped when the learning rate goes under the threshold of 10e-5
            # if lr < learning_threshold:
            #     break
            
            # # at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
            # if dev_acc > prev_acc:
            #     lr /= learning_decay
            # prev_acc = dev_acc

            # train_loss.backward()
            # optimizer.step()

    torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{model_name}.pth'))

if __name__ == '__main__':
    train()
