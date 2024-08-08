# Pending work
Can't figure out why the generated words don't match from the video. Everything looks the same. Figure it out.

# Things I would not like to forget

## Multinomial has replacement set to False by default
Always set
> torch.multinomial(..,..,replacement = True)


## Broadcasting
Semantics here, just search for broadcasting semantics
`https://pytorch.org/docs/stable/notes/broadcasting.html`

Thumb rules
> Two tensors are “broadcastable” if the following rules hold: \
    - Each tensor has at least one dimension. \
    -  **When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist** 

Take care which direction is non existent

## torch.tensor vs torch.Tensor
 1. They are different!
 2. tensor retains dype, Tensor convers to float
 > torch.tensor infers the dtype automatically, while torch. Tensor returns a torch.FloatTensor. I would recommend to stick to torch.tensor, which also has arguments like dtype, if you would like to change the type.