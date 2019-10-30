import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
% matplotlib inline
% config InlineBackend.figure_format = 'retina'

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

# making batches
int_text = [i for i in range(10)]
print('Input: ', int_text)
idx=5 # word index of interest

from get_target import get_target
target = get_target(int_text, idx=idx, window_size=5)
#print('Target: ', target)

# generating batches

from get_batches import get_batches
int_text = [i for i in range(20)]
x, y = next(get_batches(int_text, batch_size=4, window_size=5))
#print('x\n', x)
#print('y\n', y)

# define and train the SkipGram model
from SkipGram import SkipGram

# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 300 # can be changed

from train import train
train(model, vocab_to_int, embedding_dim, train_words)

# visualize the word vectors
# getting embedding from the embedding layer of our model, by name
embeddings = model.embed.weight.to('cpu').data.numpy()
viz_words = 600
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16,16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx,0], embed_tsne[idx,1]), alpha = 0.7)
