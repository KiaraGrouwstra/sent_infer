# from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import torch

# Set PATHs
PATH_TO_SENTEVAL = '../SentEval/'
PATH_TO_VEC = '.vector_cache/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1
    #
    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    #
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i
    #
    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}
    #
    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')
    #
    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    #
    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)
    #
    embeddings = np.vstack(embeddings)
    return embeddings


def sent_eval(data_path, encoder):
    # Set params for SentEval
    params_senteval = {
        'task_path': data_path,
        'usepytorch': True,
        'kfold': 5,
        'infersent': encoder,
    }
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                    'tenacity': 3, 'epoch_size': 2}
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    cpu_tasks = [
        'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    ]
    gpu_tasks = [
        'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
        'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'Length',
        'WordContent', 'Depth', 'TopConstituents',
        'BigramShift', 'Tense', 'SubjNumber',
        'ObjNumber', 'OddManOut', 'CoordinationInversion',
    ]
    transfer_tasks = [
        # error:
        # 'MR', 'CR', 'MPQA',
        *cpu_tasks
    ] + (gpu_tasks if torch.cuda.is_available() else [])
    results = se.eval(transfer_tasks)
    return results

def senteval_metrics(results):
    macros = []
    micros = []
    total = sum([result['nsamples'] for result in results.values()])
    for task, result in results.items():
        acc = result['devacc']
        n = result['nsamples']
        macros.append(acc)
        micros.append(acc * n / total)  # TODO: check
    # macro metric: average of dev accuracies, equal weight to each class
    macro = np.mean(macros)
    # micro metric: sum of dev accuracies, weighted by number of dev samples; equal weight to each per-document classification decision
    micro = np.sum(micros)
    return (micro, macro)
