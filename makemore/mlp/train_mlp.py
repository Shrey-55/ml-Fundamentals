import torch
import torch.nn.functional as F

words = open('names.txt','r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = 27

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
g = torch.Generator().manual_seed(123)
C = torch.randn((vocab_size,2), requires_grad= True) # The trainable encoding we talked about
W1 = torch.randn((6,100), requires_grad=True) # Size of weight of each node should be same as the input size
b1 = torch.randn(100)
W2 = torch.randn((100,vocab_size)) # Since output size is vocab size here
b2 = torch.randn(vocab_size)
parameters = [C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad = True

"""
    emb@W is _,100
    b is 100
    shifting to left till we can gives
    b is  1, 100
    This is what we want, one number - bias of that neuron, add to only that neuron
    Correct
"""

#Forward pass

emb = C[X] # Yes, the indexing in pytorch is amazing
h = torch.tanh(emb.view(-1,6)@W1 + b1) # Good practice to verify broadcasting
logits = h@W2 + b2
# counts = logits.exp()
# prob = counts/ counts.sum(1,keepdims=True)
# loss = -prob[torch.arange(32),Y].log().mean()
loss = F.cross_entropy(logits, Y)

#Backward pass
for p in parameters:
    p.grad = None



