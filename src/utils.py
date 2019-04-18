import itertools
import functools

# https://docs.python.org/3/library/itertools.html
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return list(zip(a, b))

# https://stackoverflow.com/a/30088323/1502035
def intersperse(e, l):    
    return list(itertools.chain(*[(i, e) for i in l]))[0:-1]

# # https://stackoverflow.com/a/37557813/1502035
# compose = lambda F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
# pipe =    lambda F: functools.reduce(lambda f, g: lambda x: f(g(x)), reversed(F))

def invert_idxs(idxs):
    rng = list(range(len(idxs)))
    inv = dict(zip(idxs, rng))
    return [inv[i] for i in rng]
