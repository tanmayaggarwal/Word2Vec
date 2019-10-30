import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils
from get_target import get_target
#% matplotlib inline
#% config InlineBackend.figure_format = 'retina'

def get_batches(words, batch_size, window_size=5):
    # create a generator of word batches as a tuple (inputs, targets)
    # returns batches of input and target data using the get_target function

    n_batches = len(words)//batch_size

    # only full batches
    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            y.extend([batch_x]*len(batch_y))
        yield x, y
