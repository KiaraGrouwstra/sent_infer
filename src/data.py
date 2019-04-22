from anycache import anycache
import torch.nn as nn
from torchtext.datasets import SNLI
from torchtext.vocab import GloVe
from torchtext.data import Field
from nltk.tokenize import TreebankWordTokenizer
from utils import *
from timer_cm import Timer
from joblib import Memory
from timeit import default_timer as timer

# constants
GLOVE_DIMS = 300  # glove embedding dimensions
MEMORY = Memory(location='cache/', verbose=0)

# @MEMORY.cache  # 30s...
@anycache(cachedir='/tmp/glove.cache')  # 10s :)
def get_glove():
    return GloVe(dim=GLOVE_DIMS)

# super fast, probably doesn't even need caching!
# @anycache(cachedir='/tmp/embeds.cache')
@MEMORY.cache
def get_embeds(vectors):
    embeddings = nn.Embedding.from_pretrained(vectors)
    embeddings.requires_grad = False
    return embeddings

def get_snli(text_field, label_field):
    # also filters out unknown label '-'! :)
    return SNLI.splits(text_field, label_field)

def get_data():
    glove = get_glove()
    tokenizer = TreebankWordTokenizer().tokenize
    text_field = Field(sequential=True, tokenize=tokenizer, include_lengths=True, lower=True, use_vocab=True)
    label_field = Field(sequential=False, pad_token=None, unk_token=None, is_target=True, use_vocab=True)
    with Timer('snli') as timer:
        print('snli{')
        splits = get_snli(text_field, label_field)
        print('}')

    text_field.build_vocab(*splits, vectors=glove)
    label_field.build_vocab(*splits)
    text_vocab = text_field.vocab
    label_vocab = label_field.vocab

    text_embeds  = get_embeds(text_vocab.vectors)
    # snli = [pick_samples(ds, n=100) for ds in splits]  # TODO: comment
    snli = splits

    return (snli, text_field, label_vocab, text_embeds)
