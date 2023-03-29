import math
import pandas as pd
import numpy as np
import torch

with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# View length and sample of dataset
print(f"Length of dataset: {len(text)}")
print(f"Sample of dataset: \n{text[:1000]}")

# View characters and length of characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"All {vocab_size} characters in data set:{''.join(chars)}")

# Create sample encoder and decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]  # encoder - takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder - takes integers, outputs string

# Test encoder and decoder
message = "Hello world"
encoded_message = encode(message)
decoded_message = decode(encoded_message)

print(f" message: {message} \n encoded message: {encoded_message} \n decoded message: {decoded_message}")

# Encode dataset as a torch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"tensor shape: {data.shape} \ntensor dtype: {data.dtype}")
print(f"Sample of tensor: \n{data[:20]}")

# Create train test split
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# Sample block size
block_size = 8
print(train_data[:block_size + 1])

# Generate visual example of training process, model should be used to all combinations to predict target
x = train_data[:block_size]
y = train_data[1:block_size + 1]

print("Visual example of training process")
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"When input is {context} the target is: {target}")

# Integrate block size into train
torch.manual_seed(42)

batch_size = 4  # independent sequences to process in parallel
block_size = 8  # maximum context for prediction


def get_batch(train, batch_size, block_size):
    # generate small batch of data for inputs x and target y
    data = train_data if train else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch(train=True, batch_size=batch_size, block_size=block_size)

print(f"inputs: \n{xb.shape} \n{xb}")
print(f"targets: \n{yb.shape} \n{yb}")

print("="*10)

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} target is {target}")


import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            # reshape to be compatible with cross entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # calculate loss for predictions
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # while history is unimportant for bigram model, this generation code will be used later with a different model
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logit, loss = m(xb, yb)
print(logit.shape)
print(f"loss is: {loss}, should be {-math.log(1/65)}")

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# Train model
# Create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


batch_size = 32
block_size = 8
for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch(train=True, batch_size=batch_size, block_size=block_size)

    # evaluate the loss
    logits, loss = m(xb, yb)  # evaluate loss
    optimizer.zero_grad(set_to_none=True)  # zero gradients from previous step
    loss.backward()  # get gradients for all parameters
    optimizer.step()  # update parameters

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
