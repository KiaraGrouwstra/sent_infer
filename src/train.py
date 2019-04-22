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
from utils import accuracy  # , oh_encode
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
MEMORY = Memory(cachedir="cache/", verbose=1)

# # number of words including padding, max sentence length in SNLI
# # max([max([max(l) for l in o.values()]) for o in snli_lengths.values()])
# words_length = 82
# # TODO: ^ where else to put this?

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

def parse_flags():
    '''returns: argparse Namespace. use var() for dict.'''
    parser = argparse.ArgumentParser()
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
    #                     # ^ value: https://piazza.com/class/jttyhskqe0165f?cid=21
    # parser.add_argument('--learning_decay', type = int, default = 5,
    #                     help='by what to divide the LR when accuracy improves, default 5')
    # parser.add_argument('--learning_threshold', type = int, default = 1e-5,
    #                     help='at which learning rate to stop the experiment, default 1e-5')
    # parser.add_argument('--optimizer_type', type = str, default = 'SGD',
    #                     help='optimizer, default SGD, also supports adam, adagrad, rmsprop, adadelta')
    # parser.add_argument('--encoder_type', type = str, default = 'baseline',
    #                     help='encoder, default BoW baseline, also supports lstm, bilstm, maxlstm')
    parser.add_argument('--data_dir', type = str, default = 'results/',
                        help='Directory for storing input data')
    parser.add_argument('--auto', action='store_true',
                        help='automagically optimize hyperparameters using an evolutionary algorithm')
    for conf in [
        {'dest': 'learning_rate', 'type': float, 'required': False, 'default': 0.1, 'help': 'Learning rate'},
        {'dest': 'max_epochs', 'type': int, 'default': 500, 'help': 'Number of epochs to run trainer.'},
        {'dest': 'batch_size', 'type': int, 'default': 64, 'help': 'Batch size to run trainer.'},
        {'dest': 'eval_freq', 'type': int, 'default': 25, 'help': 'Frequency of evaluation on the test set'},
        {'dest': 'weight_decay', 'type': float, 'default': 1e-3, 'help': 'weight decay used in the optimizer, default 1e-3'},
        {'dest': 'learning_decay', 'type': int, 'default': 5, 'help': 'by what to divide the LR when accuracy improves, default 5'},
        {'dest': 'learning_threshold', 'type': int, 'default': 1e-5, 'help': 'at which learning rate to stop the experiment, default 1e-5'},
        {'dest': 'optimizer_type', 'type': str, 'default': 'SGD', 'help': 'optimizer, default SGD, also supports adam, adagrad, rmsprop, adadelta'},
        {'dest': 'encoder_type', 'type': str, 'default': 'baseline', 'help': 'encoder, default BoW baseline, also supports lstm, bilstm, maxlstm'},
        # {'dest': 'data_dir', 'type': str, 'default': 'results/', 'help': 'Directory for storing input data'},
        {'dest': 'auto', 'action': 'store_true', 'help': 'automagically optimize hyperparameters using an evolutionary algorithm'},
    ]:
        # parser.add_argument(**conf)
        parser._add_action(argparse.Action(**conf, option_strings=[f'--{conf['dest']}']))
    flags, unparsed = parser.parse_known_args()
    return flags

# @MEMORY.cache  # 30s...
@anycache(cachedir='/tmp/glove.cache')
def get_glove():
    return GloVe(dim=GLOVE_DIMS)

# def get_snli():
#     return SNLI.splits(text_field, label_field)

# @anycache(cachedir='/tmp/embeds.cache')
@MEMORY.cache
def get_embeds(vectors):
    embeddings = nn.Embedding.from_pretrained(vectors)
    embeddings.requires_grad = False
    return embeddings

@MEMORY.cache
def get_snli(text_field, label_field):
    # also filters out unknown label '-'! :)
    return SNLI.splits(text_field, label_field)

# @anycache(cachedir='/tmp/both.cache')
@MEMORY.cache
def get_data():
    '''returns: (train, dev, test)'''
    with Timer('glove') as timer:
        print('glove{')
        # TODO: fix cache
        # glove = GloVe(dim=GLOVE_DIMS)
        glove = get_glove()
        print('}')
    tokenizer = TreebankWordTokenizer().tokenize
    text_field = Field(sequential=True, tokenize=tokenizer, include_lengths=True, lower=True)
    # TODO: investigate these
    # , stop_words={}, fix_length=None, init_token='<s>', eos_token='</s>', preprocessing=None, postprocessing=None
    label_field = Field(sequential=False, pad_token=None, unk_token=None, is_target=True)
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
    # label_field.build_vocab(test)
    with Timer('embeddings') as timer:
        print('embeddings{')
        # embeddings = nn.Embedding.from_pretrained(text_field.vocab.vectors)
        # embeddings.requires_grad = False
        embeddings = get_embeds(text_field.vocab.vectors)
        print('}')
    # snli = {'dev': dev, 'train': train, 'test': test}
    # snli = {k: pick_samples(v, n=100) for k, v in snli.items()}  # TODO: comment, flip sample/filter order
    # snli = {k: filter_samples(v, strip_unknown) for k, v in snli.items()}
    snli = [pick_samples(ds, n=100) for ds in splits]
    return (snli, embeddings)

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

#def train):
# constants
flags = parse_flags()
flag_keys = ['learning_rate', 'max_epochs', 'batch_size', 'eval_freq', 'weight_decay', 'learning_decay', 'learning_threshold', 'optimizer_type', 'encoder_type']
(lr, max_epochs, batch_size, eval_freq, weight_decay, learning_decay, learning_threshold, optim, enc_type) = itemgetter(*flag_keys)(vars(flags))
(n_dim, words_dim, embedding_dim) = range(3)

# encoder
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

# model
loss_fn = torch.nn.CrossEntropyLoss()
model = Model(GLOVE_DIMS, encoder)
model.to(device)
pars = model.parameters()

# optimizer
optim_pars = {'params': pars, 'lr': lr, 'weight_decay': weight_decay}
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

# iterate
prev_acc = 0.0
# (splits, embeddings) = get_data()
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

    # TODO: cut out into eval/infer files, see README
    (loss, acc) = eval_dataset(model, test, batch_size)
    # vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
    stats = {
        'optimizer':  optim,
        'test_acc':   acc,
        'test_loss':  loss,
        'learning_rate': lr,
    }
    print(yaml.dump({k: round(i, 3) if isinstance(i, float) else i for k, i in stats.items()}))
    w.add_scalars('metrics', stats)  # TODO: , epoch?

# # SentEval: https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
# results = sent_eval(data_path).items()
# for task, result in results:
#     w.add_scalars(f'tasks/{task}', result)  # , epoch?
# (micro, macro) = senteval_metrics(results)
# metrics = {
#     'micro': micro,
#     'macro': macro,
# }
# w.add_scalars('senteval/metrics', metrics)  # , epoch?

# w.export_scalars_to_json('./scalars.json')
return loss

def optimize():
    # TODO: investigate trixi: https://github.com/MIC-DKFZ/trixi/blob/master/examples/pytorch_experiment.ipynb
    emins = lambda n: list(reversed([10**-(i+1) for i in range(n)]))

    options = {
        'dnn_hidden_units': ['10', '100', '200,200', '400', '25,50,50,50,50,25'],
        'learning_rate': emins(5),
        'max_steps': [500, 1000, 1500, 2500],
        'batch_size': [25, 50, 100, 250],
        'optimizer': ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta'],
        'weight_decay': [0, *emins(5)],
    }

    def eval_loss(dct):
        FLAGS = argparse.Namespace(**dct)
        # ^ does this work to be in scope for train_mlp_pytorch?
        print_flags()
        return train()

    run_ea = get_ea(options, eval_loss)
    best_member, best_loss = run_ea(n=8, rounds=20)

# if __name__ == '__main__':
#     if FLAGS.auto:
#         optimize()
#     else:
#         # Run the training operation
#         train()
