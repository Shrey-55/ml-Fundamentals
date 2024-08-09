import torch
import torch.nn.functional as F

words = open('names.txt','r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

## Making dataset for MLP

context_len = 3

X, Y = [], []
for w in words[:5]:
    print (w)
    context = [0] * context_len
    for ch in w+'.':
        ix = stoi[ch]
        X. append (context)
        Y.append (ix)
        print(''.join(itos[i] for i in context),'--->',itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
"""
    X.shape is [32,3] when 5 words
    or [num_example, context_len]

    Now we want to represent each input as a number, which we did using
    the one_hot encoding, when the context lenght was 1

    Using one_hot encoding now would be very inefficient as we are dealing with a chunk of characters
    instead of just one
    So, the idea is to encode each letter in a 2 dimensional(earlier we used a vector of size 27 for each letter)
    trainable vector and learn the encoding too

"""

C = torch.randn((27,2)) # The trainable encoding we talked about
emb = C[X] # Yes, the indexing in pytorch is amazing

