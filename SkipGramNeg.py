
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#% matplotlib inline
#% config InlineBackend.figure_format = 'retina'

# to make the training more efficient, we can approximate the loss from the softmax layer
# by only updating a small subset of all weights at once. This is called negative sampling.

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist

        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        # Initialize both embedding tables with uniform distribution
        nn.init.uniform_(self.in_embed.weight, -1.0, 1.0)
        nn.init.uniform_(self.out_embed.weight, -1.0, 1.0)

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self, batch_size, n_samples, model):
        # generate noise vectors with shape (batch_size, n_samples, n_embed)
        if self.noise_dist is None:
            # sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist

        noise_sample = batch_size*n_samples
        # sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist, noise_sample, replacement=True)

        device = "cuda" if model.out_embed.weight.is_cuda else "cpu"
        noise_words = noise_words.to(device)

        noise_vectors = self.out_embed(noise_words)

        noise_vectors = noise_vectors.view(batch_size, n_samples, self.n_embed)

        return noise_vectors
