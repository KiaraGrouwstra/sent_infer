from utils import *

def test_pairwise():
    assert pairwise([1,2,3]) == [(1,2),(2,3)]

def test_intersperse():
    assert intersperse(2, [1,3]) == [1,2,3]

def test_invert_idxs():
    assert invert_idxs([2,0,1]) == [1,2,0]

def test_accuracy():
    assert accuracy(torch.Tensor([[0, 1], [1, 0]]), torch.Tensor([[0, 1], [1, 0]])) == 1.0

# def test_pick_samples():
#     assert pick_samples(ds, 10).size == 10

# def test_filter_samples():
#     assert len(filter_samples(ds, lambda: False)) == 0

# def test_unpack_tokens():
#     # assert unpack_tokens(tpl)

# def test_batch_cols():
#     # assert batch_cols(batch)

def test_prep_torch():
    assert prep_torch()
