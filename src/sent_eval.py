import io
import sys
import numpy as np
import data
import os
import logging

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = '../SentEval/'
# path to the NLP datasets 
# PATH_TO_DATA = '../data/downstream'
# path to glove embeddings
PATH_TO_VEC = '.vector_cache/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# start of functions gracefully borrowed from https://github.com/facebookresearch/SentEval/blob/master/examples/bow.py

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec

# sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc
def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

# transforms a batch of text sentences into sentence embeddings
def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

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

    embeddings = np.vstack(embeddings)
    return embeddings

# end of borrowed functions

def sent_eval(data_path, encoder):
    params_senteval = {
        'task_path': data_path,
        # 'task_path': os.getcwd(),
        'usepytorch': True,
        'kfold': 10,
        'classifier': {
            'batch_size': 64,
            'optim': 'adam', 
            'tenacity': 3,
            'epoch_size': 2,
            'nhid': 0,
        },
        'infersent': encoder,
    }
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # SICKRelatedness (Sick-R) needs torch cuda
    # STS14 lacks ndev/devacc keys, seems split up
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment']
    results = se.eval(transfer_tasks)
    return results

def senteval_metrics(results):
    macros = []
    micros = []
    total = sum([result['ndev'] for result in results.values()])
    for task, result in results.items():
        acc = result['devacc']
        n = result['ndev']
        macros.append(acc)
        micros.append(acc * n / total)  # TODO: check
    # macro metric: average of dev accuracies, equal weight to each class
    macro = np.mean(macros)
    # micro metric: sum of dev accuracies, weighted by number of dev samples; equal weight to each per-document classification decision
    micro = np.sum(micros)
    return (micro, macro)
