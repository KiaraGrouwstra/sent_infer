import itertools
import numpy as np
import torch
from torchtext.data import Dataset

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
    return (predictions.argmax(dim=-1) == targets).float().mean() # .item()

def pick_samples(ds, n):
    examples = ds.examples[0:n]
    return Dataset(examples, ds.fields)

def unpack_tokens(tpl, text_embeds):
    (words_batch, lengths) = tpl
    return (text_embeds(words_batch), lengths)

def batch_cols(batch, text_embeds):
    prem_embeds, prem_lens = unpack_tokens(batch.premise   , text_embeds)
    hyp_embeds,   hyp_lens = unpack_tokens(batch.hypothesis, text_embeds)
    labels = batch.label
    return (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels)

def prep_torch():
    # make it deterministic for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # pytorch defaults
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    return device

def get_stats(cols, vals):
    return dict(zip(cols, [
        i.item() if isinstance(i, torch.Tensor) else
        np.asscalar(i) if isinstance(i, (np.ndarray, np.generic)) else
        i for i in vals]))
