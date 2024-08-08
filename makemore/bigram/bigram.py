
words = open('names.txt','r').read().splitlines()
# print(words[:10])
print(len(words))

## storing bigram frequency in tensor
# mapping
chars = sorted(list(set(''.join(words))))
# print(len(chars))
stoi = {s:i+1 for i,s in enumerate(chars)}
# stoi['<S>'] = 26
# stoi['<E>'] = 27
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

import torch
N = torch.zeros((27,27),dtype=torch.int32)
for w in words:
    w1 = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(w1,w1[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] +=1

# print(N[0])
# p = N[0].float()
# print(p/p.sum())

# g = torch.Generator().manual_seed(2147483647)
# ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
# itos[ix]

## Skipping the visualisation part

## Sampling
# We will use torch.generator and torch.multinomial

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

P = (N+1).float()
P/= P.sum(1,keepdim=True)

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))

log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
        