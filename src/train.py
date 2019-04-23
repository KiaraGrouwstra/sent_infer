import os
import argparse
import numpy as np
import torch
from torchtext.datasets import SNLI
from torchtext.data import BucketIterator
from tensorboardX import SummaryWriter
from utils import *
from encoders import Baseline
from model import Model
from operator import itemgetter
from tqdm import tqdm
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
    parser.add_argument('--max_epochs', type = int, default = 10,  # 500
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

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

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
        for batch in train_iter:
            (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels) = batch_cols(batch, text_embeds)
            predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
            train_loss = loss_fn(predictions, labels)
            train_acc = accuracy(predictions, labels)
            # print('train_loss', train_loss)
            # print('train_acc', train_acc)

            # evaluate on dev set and report results
            if epoch % eval_freq == 0:
                (dev_loss, dev_acc) = eval_dataset(model, dev, batch_size, loss_fn, device, text_embeds)
                # print('dev_acc', dev_acc)
                print({'dev_acc': dev_acc, 'dev_loss': dev_loss})

            # training is stopped when the learning rate goes under the threshold of 10e-5
            if lr < learning_threshold:
                break
            
            # at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
            if dev_acc > prev_acc:
                lr /= learning_decay
            prev_acc = dev_acc

            train_loss.backward()
            optimizer.step()

    state = {
        'model':      model           .state_dict(),
        'optimizer':  optimizer       .state_dict(),
        'classifier': model.classifier.state_dict(),
        'encoder':    model.encoder   .state_dict(),
    }
    checkpoint_file = os.path.join(checkpoint_path, f'{model_name}.pth')
    print(checkpoint_file)
    torch.save(state, checkpoint_file)

    # evaluate on test set
    (loss, acc) = eval_dataset(model, test, batch_size, text_embeds)
    # vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
    stats = {
        # 'optimizer':  optim,
        'test_acc':   acc,
        'test_loss':  loss,
        # 'learning_rate': lr,
    }
    print(yaml.dump({k: round(i, 3) if isinstance(i, float) else i for k, i in stats.items()}))
    # w.add_scalars('metrics', stats)

if __name__ == '__main__':
    train()
