# Things I wouldn't like to forget(again)

## Notes from the video

### Model Initialisation
1. Hockey shape appearance of loss is not good, better initialise the parameters
2. Weights shouldn't be initialised to 0
    
3. While using activations, it is important to see that it doesn't go in the parts where the activation function has 0 gradient. If that is so for a neuron for all examples, it could make the neuron dead as there is no learning as the gradient is 0. Good initialisation is imprtant
    
    1. This is where kaiming normalisation comes in
    2. But, why not just normalise the hidden state before putting it through the activation layer. This is where batch normalisation and layer normalisation come in 
    3. To get the good idea of what the learning rate should be, you can plot the 



## torch.no_grad() Decorator and Context manager

Context managers are a more general term. `with` is a good example. Read more here https://book.pythontips.com/en/latest/context_managers.html

torch.no_grad is a context manager, works as a decorator too
>   Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.

    In this mode, the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True. There is an exception! All factory functions, or functions that create a new Tensor and take a requires_grad kwarg, will NOT be affected by this mode.

    This context manager is thread local; it will not affect computation in other threads.

    Also functions as a decorator.