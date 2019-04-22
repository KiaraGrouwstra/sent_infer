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

# constants
GLOVE_DIMS = 300  # glove embedding dimensions

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = 'mean', help='model, baseline (default), lstm, bilstm, maxlstm')
    parser.add_argument('--model_name', type = str, default = 'mean', help='model name, default mean')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoint.th', help='check point_path, default checkpoint_path.th')
    parser.add_argument('--train_data_path', type = str, default = 'train.txt', help='train, default train.txt')

    # parser.add_argument('--learning_rate', type = float, default = 0.1,
    #                     help='Learning rate')
    # parser.add_argument('--max_epochs', type = int, default = 500,
    #                     help='Number of epochs to run trainer.')
    # parser.add_argument('--batch_size', type = int, default = 64,
    #                     help='Batch size to run trainer.')
    # parser.add_argument('--eval_freq', type = int, default = 25,
    #                     help='Frequency of evaluation on the test set')
    # parser.add_argument('--weight_decay', type = float, default = 1e-3,
    #                     help='weight decay used in the optimizer, default 1e-3')
    # parser.add_argument('--learning_decay', type = int, default = 5,
    #                     help='by what to divide the LR when accuracy improves, default 5')
    # parser.add_argument('--learning_threshold', type = int, default = 1e-5,
    #                     help='at which learning rate to stop the experiment, default 1e-5')
    # parser.add_argument('--optimizer_type', type = str, default = 'SGD',
    #                     help='optimizer, default SGD, also supports adam, adagrad, rmsprop, adadelta')
    # parser.add_argument('--encoder_type', type = str, default = 'baseline',
    #                     help='encoder, default BoW baseline, also supports lstm, bilstm, maxlstm')
    # parser.add_argument('--data_dir', type = str, default = 'results/',
    #                     help='Directory for storing input data')
    # parser.add_argument('--auto', action='store_true',
    #                     help='automagically optimize hyperparameters using an evolutionary algorithm')

    flags, unparsed = parser.parse_known_args()
    return flags

#def train):
# constants
flags = parse_flags()
flag_keys = ['model_type', 'model_name', 'checkpoint_path', 'train_data_path', 'learning_rate', 'max_epochs', 'batch_size', 'eval_freq', 'weight_decay', 'learning_decay', 'learning_threshold', 'optimizer_type', 'encoder_type']
(model_type, model_name, checkpoint_path, train_data_path, lr, max_epochs, batch_size, eval_freq, weight_decay, learning_decay, learning_threshold, optim, enc_type) = itemgetter(*flag_keys)(vars(flags))

model = get_model(enc_type)
pars = model.parameters()
optim_pars = {'params': pars, 'lr': lr, 'weight_decay': weight_decay}
optimizer = get_optimizer(optim, model.parameters())

# iterate
prev_acc = 0.0
# (snli, text_vocab, label_vocab, text_embeds) = get_data()
# (train, dev, test) = splits
name = encoder().__class__.__name__
with SummaryWriter(name) as w:
    for epoch in tqdm(range(max_epochs)):
        optimizer.zero_grad()

        # train
        # (train_iter,) = BucketIterator.splits(datasets=(train,), batch_sizes=[batch_size], device=device, shuffle=True)
        # for batch in tqdm(train_iter):
        for epoch in range(max_epochs):
            set_trace()
            batch = pickle.load('batch_ex.pkl')
            (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels) = batch_cols(batch)
            predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
            train_loss = loss_fn(predictions, labels)
            train_acc = accuracy(predictions, labels)

            # evaluate on dev set and report results
            if epoch % eval_freq == 0:
                time = int(epoch / eval_freq)
                start = timer()
                (dev_loss, dev_acc) = eval_dataset(model, dev, batch_size)
                end = timer()
                secs = end - start

                # vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
                stats = {k: np.asscalar(i) if isinstance(i, (np.ndarray, np.generic)) else i for k, i in {
                    'optimizer':  optim,
                    'train_acc':  train_acc,
                    'dev_acc':    dev_acc,
                    'train_loss': train_loss,
                    'dev_loss':   dev_loss,
                    'learning_rate': lr,
                    'secs': secs,
                }.items()}
                print(yaml.dump({k: round(i, 3) if isinstance(i, float) else i for k, i in stats.items()}))
                w.add_scalars('metrics', stats, time)

            # training is stopped when the learning rate goes under the threshold of 10e-5
            if lr < learning_threshold:
                break
            
            # at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
            if dev_acc > prev_acc:
                lr /= learning_decay
            prev_acc = dev_acc

            train_loss.backward()
            optimizer.step()

    df = pd.DataFrame(results, columns=cols)
    meta = {
        # 'framework': 'pytorch',
        'encoder': enc_type,
        'optimizer': optim,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    for k, v in meta.items():
        df[k] = v
    output_file = 'results.csv'  # f'{name}.csv'
    if os.path.isfile(output_file):
        df.to_csv(f'{name}.csv', header=False, mode='a')
    else:
        df.to_csv(f'{name}.csv', header=True, mode='w')
    torch.save(model.state_dict(), f'{name}.pth')
    print('done!')

# if __name__ == '__main__':
#     if FLAGS.auto:
#         optimize()
#     else:
#         # Run the training operation
#         train()
