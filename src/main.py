# import pickle
import argparse
import itertools
from functools import reduce
import numpy as np
import torch
import torch.nn
# import torchnlp
import torch.optim
# from torchnlp.datasets import snli_dataset
from torchtext.datasets import SNLI
# from torchnlp.word_to_vector import GloVe
from torchtext.vocab import GloVe
from torchnlp.encoders import LabelEncoder
import torchtext
from torchtext.data import Field
from nltk.tokenize import TreebankWordTokenizer
from tensorboardX import SummaryWriter
from utils import accuracy  # , oh_encode
from dataset import DataSet
# from sent_eval import sent_eval, senteval_metrics  # batcher, prepare
from encoders import Baseline, lstms  # uni_lstm, bi_lstm
from model import Model
from operator import itemgetter
from pdb import set_trace

unknown_key = '<unk>'  # GloVe
sets = ['train', 'dev', 'test']

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
torch.set_default_tensor_type(dtype)

def parse_flags():
    '''returns: argparse Namespace. use var() for dict.'''
    parser = argparse.ArgumentParser()
    # for conf in []:
    #     parser.add_argument(**conf)
    parser.add_argument('--learning_rate', type = float, default = 0.1,
                        help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = 500,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = 64,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type = int, default = 25,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--weight_decay', type = int, default = 0.99,
                        help='weight decay used in the optimizer, default 0.99')
    parser.add_argument('--learning_decay', type = int, default = 5,
                        help='by what to divide the LR when accuracy improves, default 5')
    parser.add_argument('--learning_threshold', type = int, default = 1e-5,
                        help='at which learning rate to stop the experiment, default 1e-5')
    parser.add_argument('--optimizer_type', type = str, default = 'SGD',
                        help='optimizer, default SGD, also supports adam, adadelta')
    parser.add_argument('--encoder_type', type = str, default = 'baseline',
                        help='encoder, default BoW baseline, also supports lstm, bilstm')
    # parser.add_argument('--data_dir', type = str, default = 'results/',
    #                     help='Directory for storing input data')
    flags, unparsed = parser.parse_known_args()
    return flags

def tokenize(snli_set):
    # TODO: does this add <s> </s> tags?
    tokenizer = TreebankWordTokenizer()
    token_set = set()
    for item in snli_set:
        for k in ['premise', 'hypothesis']:
            for token in tokenizer.tokenize(item[k]):
                token_set.add(token)
    return token_set

def get_data():
    '''returns: (train, dev, test)'''
    print('glove{')
    # TODO: fix cache
    glove = GloVe(dim=n_inputs)
    tokenizer = TreebankWordTokenizer().tokenize
    text_field = Field(sequential=True, tokenize=tokenizer, include_lengths=True, lower=True)
    # TODO: investigate these
    # , stop_words={}, fix_length=None, init_token='<s>', eos_token='</s>', preprocessing=None, postprocessing=None
    label_field = Field(sequential=False, pad_token=None, unk_token=None, is_target=True)
    print('snli{')
    train, dev, test = SNLI.splits(text_field, label_field)
    print('}')

    text_field.build_vocab(full_train_set, full_dev_set, full_test_set, vectors=glove)
    # label_field.build_vocab(snli['test'])
    embeddings = nn.Embedding.from_pretrained(text_field.vocab.vectors)
    embeddings.requires_grad = False
    # ds.filter_examples(['label'])  # filters out unknown words, so somehow whitelist in label_field.build_vocab?
    # return SNLI.splits(text_field, label_field)
    # splits = SNLI.splits(text_field, label_field)
    # return splits
    # return [
    #     item for item in splits
    #     #  if item['label'] != '-'
    # ]
    set_trace()
    data = {'dev': dev, 'train': train, 'test': test}
    return (data, embeddings)
    # TODO: filter out '-'
    # return {k: v[0:100] for k, v in {'dev': dev, 'train': train, 'test': test}.items()}
    
    # snli_dir = 'data/snli_1.0/'
    # return {k: [
    #     item for item in snli_dataset(snli_dir, **{k: True})[0:100] if item['label'] != '-'
    #     # TODO: remove [0:100]
    # ] for k in sets}

def eval_dataset(model, dataset, eval_rounds, data_keys, batch_size):
    cols = ['loss', 'acc']
    df = pd.DataFrame([], columns=cols)
    for t in range(eval_rounds):
        (labels, hyp_lens, prem_lens, hyp_embeds, prem_embeds) = itemgetter(*data_keys)(dataset.next_batch(batch_size))
        predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
        df = df.append([dict(zip(cols, [
            loss_fn( predictions, labels),
            accuracy(predictions, labels),
        ]))])
    (loss, acc) = list(df.mean())
    return (loss, acc)

#def main():
# constants
n_inputs = 300  # glove embedding dimensions
flags = parse_flags()
flag_keys = ['learning_rate', 'max_steps', 'batch_size', 'eval_freq', 'weight_decay', 'learning_decay', 'learning_threshold', 'optimizer_type', 'encoder_type']
(lr, max_steps, batch_size, eval_freq, weight_decay, learning_decay, learning_threshold, optim, enc_type) = itemgetter(*flag_keys)(vars(flags))

# data
snli = get_data()

# clean_text = lambda s: s.lower()
# token_set = tokenize(snli['train'])
# words = set(map(clean_text, token_set))
# unknown = embeddings[unknown_key]

kind_encoder = LabelEncoder(['entailment', 'contradiction', 'neutral'])

tokenizer = TreebankWordTokenizer()
# snli_labels_oh = {k: oh_encode([item['label'] for item in items], label_encoding) for k, items in snli.items()}
# snli_labels = {k: torch.IntTensor([label_encoding[item['label']] for item in items]) for k, items in snli.items()}
kinds = ['premise', 'hypothesis']
snli_tokens = {k: {kind: [tokenizer.tokenize(item[kind]) for item in items] for kind in kinds} for k, items in snli.items()}

# set_trace()
# snli_embeddings = {k: {kind: [[
#         embeddings.get(token, unknown)
#     for token in tokens] for tokens in dct[kind]] for kind in kinds} for k, dct in snli_tokens.items()}

# (n_dim, words_dim, kind_dim, embedding_dim)
# TODO: .to(device)
# prop = lambda k: lambda x: x[k]
# TODO: , requires_grad = dataset == 'train'
data = {dataset: DataSet({
    # 'labels':             torch.LongTensor([label_encoding[item['label']] for item in snli[dataset]]),
    'labels':             torch.LongTensor(kind_encoder.batch_encode([item['label'] for item in snli[dataset]])),
    # 'labels':             torch.LongTensor(kind_encoder.batch_encode(map(prop('label'), snli[dataset]))),
    'hypothesis_lengths': torch.LongTensor([len(item['hypothesis'])       for item in snli[dataset]]),
    'premise_lengths':    torch.LongTensor([len(item['premise'   ])       for item in snli[dataset]]),
    'hypothesis_embeddings': torch.FloatTensor(np.array([[embeddings.get(token, unknown).numpy() for token in tokens] for tokens in snli_tokens[dataset]['hypothesis']])),
       'premise_embeddings': torch.FloatTensor(np.array([[embeddings.get(token, unknown).numpy() for token in tokens] for tokens in snli_tokens[dataset]['premise'   ]])),
}) for dataset in sets}
data_keys = ['labels', 'hypothesis_lengths', 'premise_lengths', 'hypothesis_embeddings', 'premise_embeddings']
eval_rounds = {k: int(np.ceil(dataset._num_examples / batch_size)) for k, dataset in data.items()}

(n_dim, words_dim, embedding_dim) = range(3)  # , kind_dim

# encoder
# words_length = words_embeddings.size[words_dim]
[uni_lstm, bi_lstm] = lstms(words_length)
if enc_type == 'lstm':
    encoder = uni_lstm
elif enc_type == 'bilstm':
    encoder = bi_lstm
else:  # baseline
    encoder = lambda: Baseline(words_length, words_dim)

# model
loss_fn = torch.nn.CrossEntropyLoss()
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
for step in range(max_steps):
    optimizer.zero_grad()

    # batch
    # for chunk in torchtext.data.batch(data['train'], batch_size):  # my DataSet lacks an iterator, maybe pandas / torch.util.Dataset?
    (labels, hyp_lens, prem_lens, hyp_embeds, prem_embeds) = itemgetter(*data_keys)(data['train'].next_batch(batch_size))
    predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
    train_loss = loss_fn(predictions, labels)
    train_acc = accuracy(predictions, labels)

    # evaluate on dev set and report results
    if step % eval_freq == 0:
        # cols = ['loss', 'acc']
        # df = pd.DataFrame([], columns=cols)
        # for t in range(eval_rounds['dev']):
        #     (labels, hyp_lens, prem_lens, hyp_embeds, prem_embeds) = itemgetter(*data_keys)(data['dev'].next_batch(batch_size))
        #     predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
        #     df = df.append([dict(zip(cols, [
        #         loss_fn( predictions, labels),
        #         accuracy(predictions, labels),
        #     ]))])
        # (dev_loss, dev_acc) = list(df2.mean())
        (dev_loss, dev_acc) = eval_dataset(model, data['dev'], eval_rounds['dev'], data_keys, batch_size)

        # vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
        stats = {
            'optimizer':  optim,
            'train_acc':  train_acc,
            'dev_acc':    dev_acc,
            'train_loss': train_loss,
            'dev_loss':   dev_loss,
            'learning_rate': lr,
        }
        # print(stats)
        writer.add_scalars('metrics', stats, step)

    # training is stopped when the learning rate goes under the threshold of 10e-5
    if lr < learning_threshold:
        break
    
    # at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
    if dev_acc > prev_acc:
        lr /= learning_decay
    prev_acc = dev_acc

    train_loss.backward()
    optimizer.step()

(loss, acc) = eval_dataset(model, data['test'], eval_rounds['test'], data_keys, batch_size)
# vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
stats = {
    'optimizer':  optim,
    'test_acc':   acc,
    'test_loss':  loss,
    'learning_rate': lr,
}
print(stats)
writer.add_scalars('metrics', stats)  # TODO: , step?

# # SentEval: https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
# results = sent_eval(data_path).items()
# for task, result in results:
#     writer.add_scalars(f'tasks/{task}', result)  # , step?
# (micro, macro) = senteval_metrics(results)
# metrics = {
#     'micro': micro,
#     'macro': macro,
# }
# writer.add_scalars('senteval/metrics', metrics)  # , step?

# writer.export_scalars_to_json('./scalars.json')
writer.close()

if __name__ == '__main__':
    main()
