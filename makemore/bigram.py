import torch

words = open('names.txt','r').read().splitlines()
# print(words[:10])

## storing bigram frequency in tensor
# mapping
chars = sorted(list(set(''.join(words))))
# print(len(chars))
stoi = {s:i+1 for i,s in enumerate(chars)}
# stoi['<S>'] = 26
# stoi['<E>'] = 27
stoi['<.>'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

N = torch.zeros((27,27),dtype=torch.int32)
for w in words:
    w1 = ['<.>'] + list(w) + ['<.>']
    for ch1,ch2 in zip(w1,w1[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] +=1

# print(N)

## Skipping the visualisation part

## Sampling
# We will use torch.generator and torch.multinomial
g = torch.Generator().manual_seed(2147483647)

# Naive for loop
# ix = 0

# while True:
#     p = N[ix].float()
#     p = p/p.sum()
#     ix = torch.multinomial(p,1,replacement=True,generator=g).item()
#     c = itos[ix]
#     if c == '<.>':
#         break
#     print(c)

P = N.float()
P/= P.sum(1,keepdim=True)
for _ in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p,1,True,generator=g).item()
        c = itos[ix]
        if ix==0:
            break
        out+=[c]
    print(''.join(out))

        