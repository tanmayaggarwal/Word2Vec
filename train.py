import numpy as np
import torch
from torch import nn
import torch.optim as optim
from collections import Counter
import random
import utils
from SkipGram import SkipGram
from get_batches import get_batches
from cosine_similarity import cosine_similarity

def train(model, vocab_to_int, embedding_dim, train_words):

    model = SkipGram(len(vocab_to_int), embedding_dim).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print_every = 500
    steps = 0
    epochs = 2

    for e in range(epochs):
        # get input and target batches
        for inputs, targets in get_batches(train_words, 512):
            steps += 1
            inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)

            log_ps = model(inputs)
            loss = criterion(log_ps, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                # getting examples and similarities
                valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
                _, closest_idxs = valid_similarities.topk(6) # topk highest similarities

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                for ii, valid_idx in enumerate(valid_examples):
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                    print(int_to_vocab[valid_idx.item()]+" | " + ", ".join(closest_words))
                print("...")
    return
