import numpy as np

class DataSet(object):
  """
  Utility class to handle dataset structure.
  """

  def __init__(self, stuff):
    """
    Builds dataset
    """
    lengths = list(set([tensor.shape[0] for tensor in stuff.values()]))
    assert len(lengths) == 1

    self._num_examples = lengths[0]
    self._stuff = stuff
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def stuff(self):
    return self._stuff

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """
    Return the next `batch_size` examples from this data set.
    Args:
      batch_size: Batch size.
    """
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      self._epochs_completed += 1

      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._stuff = {k: v[perm] for k, v in self._stuff.items()}

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return {k: v[start:end] for k, v in self._stuff.items()}
