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
MEMORY = Memory(cachedir="cache/", verbose=1)


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
    return (predictions.argmax(dim=-1) == targets.argmax(dim=-1)).type(dtype).mean().detach().data.cpu().item()

def get_encoder(enc_type):
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

def get_model(enc_type):
    encoder = get_encoder(enc_type)
    loss_fn = torch.nn.CrossEntropyLoss()
    model = Model(GLOVE_DIMS, encoder)
    device = prep_torch()
    model.to(device)
    return model
