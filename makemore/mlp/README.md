# Things I wouldn't like to forget

## Indexing in pytorch
While indexing a tensor in pytorch with a list and a tensor, it might work differently. See below for example
```python
C[torch.tensor([[1,2],[2,3]])]

tensor([[[ 0.1035,  0.1628],
         [-1.4500, -0.3615]],

        [[-1.4500, -0.3615],
         [ 1.2268, -1.3771]]])
```
But

```python
C[[[1,2],[2,3]]] == [C[1,2],C[2,3]]
```


## torch.view() 
torch.view() is much more efficient than 'torch.cat(torch.unbind(blah,1),1)
torch.view doesn't change storage but only the stride, length etc of the tensor is change making it much more efficient
Just a pytorch internal mechanishm thing