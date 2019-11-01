import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils
from SkipGramNeg import SkipGramNeg
from get_batches import get_batches
from cosine_similarity import cosine_similarity
from NegativeSamplingLoss import NegativeSamplingLoss

def train_Neg(vocab_to_int, int_to_vocab, embedding_dim, noise_dist, train_words, device):

    model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)

    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print_every = 1500
    steps = 0
    epochs = 2
    #print(len(train_words))
    for e in range(epochs):
        # get input and target batches
        for input_words, target_words in get_batches(train_words, 512):
            #print(len(input_words))
            steps += 1
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
            inputs, targets = inputs.to(device), targets.to(device)

            # input, output, and noise vectors
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(inputs.shape[0], 5, model)

            loss = criterion(input_vectors, output_vectors, noise_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                print("Epoch: {}/{}".format(e+1, epochs))
                print("Loss: ", loss.item()) # avg batch loss at this point in training
                valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                print("...\n")
    return model
