import sys
import numpy as np
import data

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = '../'
# path to the NLP datasets 
PATH_TO_DATA = '../data/downstream'
# path to glove embeddings
PATH_TO_VEC = '../pretrained/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# sees the whole dataset of each task and can thus construct the word vocabulary, the dictionary of word vectors etc
def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    _, params.word2id = data.create_dictionary(samples)
    # load glove/word2vec format 
    params.word_vec = data.get_wordvec(PATH_TO_VEC, params.word2id)
    # dimensionality of glove embeddings
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
        # the format of a sentence is a lists of words (tokenized and lowercased)
        for word in sent:
            if word in params.word_vec:
                # [number of words, embedding dimensionality]
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            # [number of words, embedding dimensionality]
            sentvec.append(vec)
        # average of word embeddings for sentence representation
        # [embedding dimansionality]
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)
    # [batch size, embedding dimensionality]
    embeddings = np.vstack(embeddings)
    return embeddings

# TODO: how does this even use our model?
def sent_eval(data_path):
    params_senteval = {'task_path': data_path, 'usepytorch': True, 'kfold': 10}
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
        micros.append(acc * n / total)  # TODO: is this weighting correct?
    # macro metric: average of dev accuracies, equal weight to each class
    macro = np.mean(macros)
    # micro metric: sum of dev accuracies, weighted by number of dev samples; equal weight to each per-document classification decision
    micro = np.sum(micros)
    return (micro, macro)
