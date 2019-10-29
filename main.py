import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils
#% matplotlib inline
#% config InlineBackend.figure_format = 'retina'

file_path = 'data/text8'

from load_data import load_data
text = load_data(file_path)

from pre_process import pre_process
words = pre_process(text)

# print stats about the word dataset
print("Total words in text: {}".format(len(words)))
print("Unique words: {}".format(len(set(words))))

# creating two dictionaries to convert words to integers and vice versa

vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]
# print(int_words[:30])

# subsampling the data to remove words that show up often (e.g., the, of, for) but provide little to no context to the nearby words
# subsampling helps remove the noise in the data and get faster training and better representation

from subsampling import subsampling
train_words = subsampling(int_words, word_counts)
