## Read the file
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print("Length of dataset: ", len(text))

# print("First 1000 characters of the dataset:\n", text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(vocab_size)

## Encode and decoder
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hello shrey"))
print(decode(encode("hello shrey")))


## Convert to torch
import torch 
data = torch.tensor(encode(text),dtype = torch.long)
print(data.shape,data.type)
print(data[:100])

## Train val split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

## Set the context
block_size = 8
batch_size = 4
torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size,(block_size,))
    x = torch.stack([data[i:i+block_size] for  i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for  i in ix])
    return x,y

xb,yb = get_batch("train")

print(xb)

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    
    def forward(self,idx,targets):
        logits = self.token_embedding_table(idx)

        return logits
m = BigramLanguageModel(vocab_size)
out = m(xb,yb)
print(out.shape)

print(xb)
print
