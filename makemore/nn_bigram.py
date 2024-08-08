import torch
import torch.nn.functional as F
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


xs, ys = [], []
for w in words:
    w1 = ['<.>'] + list(w) + ['<.>']
    for ch1,ch2 in zip(w1,w1[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

X = torch.tensor(xs)
Y = torch.tensor(ys)
n = len(xs)

X_enc = F.one_hot(X).float()
print(X_enc.shape,n)

g = torch.Generator().manual_seed(214748367)
W = torch.randn((27,27), generator = g, requires_grad = True)

for i in range(100):
# Forward pass
    logits = X_enc @ W
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdims=True)
    # print(prob.shape)
    loss = -prob[torch.arange(n),ys].log().mean() + 0.02*(W**2).mean()
    print(loss)

    #Backward pass
    W.grad = None
    loss.backward()

    W.data += -50*W.grad

for _ in range(10):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]),num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(prob,1,True,generator=g).item()
        c = itos[ix]
        if ix==0:
            break
        out+=[c]
    print(''.join(out))






    