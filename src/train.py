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
import yaml

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

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    loss_fn = torch.nn.CrossEntropyLoss()
    (model, optimizer) = make_model(model_type, lr, weight_decay, optim, batch_size)
    # iterate
    prev_acc = 0.0
    device = prep_torch()
    (snli, text_field, label_vocab, text_embeds) = get_data()
    (train, dev, test) = snli
    cols = ['loss', 'acc']
    stage_csv_file = lambda stage: os.path.join(checkpoint_path, f'{model_name}_{stage}.csv')
    with SummaryWriter(model_name) as w:
        for epoch in tqdm(range(max_epochs)):

            stage = 'train'
            csv_file = stage_csv_file(stage)
            (train_loss, train_acc, train_df) = eval_dataset(model, train, batch_size, loss_fn, device, text_embeds, optimizer, stage, csv_file, True)

            # evaluate on dev set and report results
            if epoch % eval_freq == 0:
                stage = 'dev'
                csv_file = stage_csv_file(stage)
                (dev_loss, dev_acc, dev_df) =   eval_dataset(model, dev,   batch_size, loss_fn, device, text_embeds, optimizer, stage, csv_file, False)

                vals = [train_loss, train_acc]
                stats = get_stats(cols, vals)
                w.add_scalars(f'metrics/train', stats, int(epoch / eval_freq))

                vals = [dev_loss, dev_acc]
                stats = get_stats(cols, vals)
                w.add_scalars(f'metrics/dev', stats, int(epoch / eval_freq))

            # training is stopped when the learning rate goes under the threshold of 10e-5
            if lr < learning_threshold:
                break
            
            # at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
            if dev_acc > prev_acc:
                lr /= learning_decay
            prev_acc = dev_acc

        # save model
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
        stage = 'test'
        csv_file = stage_csv_file(stage)
        (loss, acc, test_df) = eval_dataset(model, test, batch_size, loss_fn, device, text_embeds, optimizer, stage, csv_file, False)
        
        # save result csv
        sub_dfs = {'train': train_df, 'dev': dev_df, 'test': test_df}
        stages = sub_dfs.keys()
        df = pd.DataFrame([], columns=[f'{stage}-{metric}' for stage in stages for metric in cols])
        for stage in stages:
            for metric in cols:
                col = f'{stage}-{metric}'
                sub_df = sub_dfs[stage]
                df[col] = sub_df[metric]
        # csv_file = stage_csv_file('all')
        csv_file = os.path.join(checkpoint_path, f'{model_name}.csv')
        df.to_csv(csv_file)
        csv_file = os.path.join(checkpoint_path, 'results.csv')
        if os.path.isfile(csv_file):
            df.to_csv(csv_file, header=False, mode='a')
        else:
            df.to_csv(csv_file, header=True, mode='w')

        # total = len(test.examples)
        # labels = test.label
        # # predictions = ?
        # c_correct = len(np.where(predictions['contradiction'] == labels['contradiction'])[0])
        # e_correct = len(np.where(predictions['entailment'] == labels['entailment'])[0])
        # n_correct = len(np.where(predictions['neutral'] == labels['neutral'])[0])
        # micro = ((c_correct / len(predictions['contradiction'])) +
        #         (e_correct / len(predictions['entailment'])) +
        #         (n_correct / len(predictions['neutral']))) / 3
        # macro = (c_correct + e_correct + n_correct) / total

if __name__ == '__main__':
    train()
