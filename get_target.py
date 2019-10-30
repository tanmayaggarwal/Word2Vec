import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils
#% matplotlib inline
#% config InlineBackend.figure_format = 'retina'

def get_target(words, idx, window_size=5):
    # get a list of words in a window around an index
    random_number = np.random.randint(1,window_size+1)
    min_idx = idx - random_number if (idx - random_number) > 0 else 0
    max_idx = idx + random_number
    target_words = words[min_idx:idx] + words[idx+1:max_idx+1]
    return target_words
