import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils

# subsampling the data to remove words that show up often (e.g., the, of, for) but provide little to no context to the nearby words
# subsampling helps remove the noise in the data and get faster training and better representation
def subsampling(int_words, word_counts):
    threshold = 1e-5
    word_counts = Counter(int_words)
    # print(list(word_counts.items())[0])

    total_count = len(int_words)
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}

    # discard some frequent words
    # create a new list of words for training
    train_words = [word for word in int_words if random.random() < (1-p_drop[word])]

    return train_words
