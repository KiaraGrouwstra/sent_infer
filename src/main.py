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

dtype = torch.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DATA_DIR_DEFAULT = 'results/'
LEARNING_RATE_DEFAULT = 0.1
MAX_STEPS_DEFAULT = 500
BATCH_SIZE_DEFAULT = 64
EVAL_FREQ_DEFAULT = 25
WEIGHT_DECAY_DEFAULT = 0.99
LEARNING_DECAY_DEFAULT = 5
LEARNING_THRESHOLD_DEFAULT = 1e-5
OPTIMIZER_DEFAULT = 'SGD'

# Set the random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def parse_flags():
    '''returns: FLAGS, unparsed'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--weight_decay', type = int, default = WEIGHT_DECAY_DEFAULT,
                        help='weight decay used in the optimizer, default 0.99')
    parser.add_argument('--learning_decay', type = int, default = LEARNING_DECAY_DEFAULT,
                        help='by what to divide the LR when accuracy improves, default 5')
    parser.add_argument('--learning_threshold', type = int, default = LEARNING_THRESHOLD_DEFAULT,
                        help='at which learning rate to stop the experiment, default 1e-5')
    parser.add_argument('--optimizer_type', type = str, default = OPTIMIZER_DEFAULT,
                        help='optimizer, default SGD, also supports adam, adadelta')
    # parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
    #                     help='Directory for storing input data')
    return parser.parse_known_args()

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

#def main():
# constants
FLAGS, unparsed = parse_flags()
n_inputs = 300  # glove embedding dimensions
lr = FLAGS.learning_rate
weight_decay = FLAGS.weight_decay
optim = FLAGS.optimizer_type
batch_size = FLAGS.batch_size
writer = SummaryWriter()

# data
snli = get_data()
# words = tokenize(test)  # all
merge_sets = lambda x, y: {*x, *y}
clean_text = lambda s: s.lower()
token_sets = map(tokenize, snli.values())
word_set = reduce(merge_sets, token_sets)
words = set(map(clean_text, word_set))
embeddings = get_embeddings(words)

# encoder
# encoders = [Baseline(), uni_lstm, bi_lstm]
# words_length = words_embeddings.size[words_dim]
# [uni_lstm, bi_lstm] = lstms(words_length)
encoder = Baseline()

# model
model = Model(n_inputs, encoder)
model.to(device)
pars = model.parameters()

# optimizer
if optim == 'adadelta':
    optimizer = torch.optim.Adadelta(pars, lr=lr, weight_decay=weight_decay)
if optim == 'adam':
    optimizer = torch.optim.Adam(    pars, lr=lr, weight_decay=weight_decay)
else:  # SGD
    optimizer = torch.optim.SGD(     pars, lr=lr, weight_decay=weight_decay)

# iterate
prev_acc = 0.0
for step in range(FLAGS.max_steps):
    optimizer.zero_grad()

    # batch
    # x_train_np, y_train_np = cifar10['train'].next_batch(batch_size)
    # x_train_flat = x_train_np.reshape((batch_size, n_inputs))
    x_train_torch = torch.from_numpy(x_train_flat)     .to(device)
    y_train_torch = torch.from_numpy(y_train_np).long().to(device)
    idx_train = torch.argmax(y_train_torch, dim=-1).long()

    # results
    train_predictions = model.forward(x_train_torch, lengths_train)
    train_loss = ce(train_predictions, idx_train)
    train_acc = accuracy(train_predictions, idx_train)

    # evaluate
    if step % FLAGS.eval_freq == 0:
        dev_accs = []
        dev_losses = []
        test_accs = []
        test_losses = []
        for t in range(eval_rounds):
            # test_predictions = model.forward(x_test_torch, lengths_test)
            # dev
            dev_loss = ce(dev_predictions, idx_dev)
            dev_acc = accuracy(dev_predictions, idx_dev)
            dev_losses.append(dev_loss)
            dev_accs  .append(dev_acc)
            # test
            test_loss = ce(test_predictions, idx_test)
            test_acc = accuracy(test_predictions, idx_test)
            test_losses.append(test_loss)
            test_accs  .append(test_acc)

        dev_acc =  np.mean(dev_accs)
        dev_loss = np.mean(dev_losses)
        test_acc =  np.mean(test_accs)
        test_loss = np.mean(test_losses)

        # vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
        stats = {
            'optimizer':  optim,
            'train_acc':  train_acc,
            'dev_acc':    dev_acc,
            'test_acc':   test_acc,
            'train_loss': train_loss,
            'dev_loss':   dev_loss,
            'test_loss':  test_loss,
            'learning_rate': lr,
        }
        # print(stats)
        writer.add_scalars('metrics', stats, step)

    # training is stopped when the learning rate goes under the threshold of 10e-5
    if lr < FLAGS.learning_threshold:
        break
    
    # at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
    if dev_acc > prev_acc:
        lr /= FLAGS.learning_decay
    prev_acc = dev_acc

    train_loss.backward()
    optimizer.step()

# # SentEval: https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
# results = sent_eval(data_path).items()
# for task, result in results:
#     writer.add_scalars(f'tasks/{task}', result, step?)
# (micro, macro) = senteval_metrics(results)
# metrics = {
#     'micro': micro,
#     'macro': macro,
# }
# writer.add_scalars('senteval/metrics', metrics, step?)

# writer.export_scalars_to_json('./scalars.json')
writer.close()

if __name__ == '__main__':
    main()
