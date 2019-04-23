import torch
import torch.optim
from torchtext.data import BucketIterator
from utils import *
from encoders import Baseline, lstms
from model import Model
import pandas as pd
from tqdm import tqdm
import yaml

# constants
GLOVE_DIMS = 300  # glove embedding dimensions

def get_encoder(enc_type):
    input_size = GLOVE_DIMS
    [uni_lstm, bi_lstm, max_lstm] = lstms(input_size)
    if enc_type == 'lstm':
        encoder = uni_lstm
    elif enc_type == 'bilstm':
        encoder = bi_lstm
    elif enc_type == 'maxlstm':
        encoder = max_lstm
    else:  # baseline
        encoder = Baseline()  # words_length
    return encoder

def eval_dataset(model, dataset, batch_size, loss_fn, device, text_embeds, optimizer, stage, update_grad=False):
    cols = ['loss', 'acc']
    df = pd.DataFrame([], columns=cols)
    (iterator,) = BucketIterator.splits(datasets=(dataset,), batch_sizes=[batch_size], device=device, shuffle=True)
    for batch in iterator:
        (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels) = batch_cols(batch, text_embeds)
        predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
        loss = loss_fn(predictions, labels)
        acc = accuracy(predictions, labels)
        vals = [loss, acc]
        stats = get_stats(cols, vals)
        print(yaml.dump({stage: {k: round(i, 3) if isinstance(i, float) else i for k, i in stats.items()}}))
        df = df.append([dict(zip(cols, [loss, acc]))])
        if update_grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

def get_model(enc_type, batch_size):
    encoder = get_encoder(enc_type)
    model = Model(batch_size, encoder)
    device = prep_torch()
    model.to(device)
    return model

def make_model(model_type, lr, weight_decay, optim, batch_size):
    model = get_model(model_type, batch_size)
    pars = model.parameters()
    optim_pars = {'params': pars, 'lr': lr, 'weight_decay': weight_decay}
    optimizer = get_optimizer(optim, optim_pars)
    return (model, optimizer)
