from torchtext.data import Field
from nltk.tokenize import TreebankWordTokenizer
from data import *

def test_get_glove():
    # assert 
    with Timer('glove1') as timer:
        assert get_glove()
    with Timer('glove2') as timer:
        assert get_glove()

def test_get_snli():
    assert get_snli()

# def test_get_embeds():
#     # assert get_embeds(vectors) == ?

def test_get_snli():
    tokenizer = TreebankWordTokenizer().tokenize
    text_field = Field(sequential=True, tokenize=tokenizer, include_lengths=True, lower=True)
    label_field = Field(sequential=False, pad_token=None, unk_token=None, is_target=True)
    # assert 
    with Timer('snli1') as timer:
        get_snli(text_field, label_field)
    with Timer('snli2') as timer:
        get_snli(text_field, label_field)

def test_get_data():
    # assert 
    with Timer('data1') as timer:
        (snli, text_field, label_vocab, text_embeds) = get_data()
    with Timer('data2') as timer:
        (snli, text_field, label_vocab, text_embeds) = get_data()
