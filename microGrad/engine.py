## Mircro grad from scratch
from graphcode import draw_dot
import math
class Value: #Because we want a graph

    def __init__(self,data,_children = (), _op ='', label = ''): # Underscore means the variable is private, read the readme for more

        self.data = data
        self._prev = set(_children)
        self.label = label
        self._op = _op
        self._backward = lambda : None
        self.grad = 0.0

    def __repr__(self): # Double 
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)

        out = Value(self.data + other.data, (self,other),'+')
        
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad

        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data + out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
        return self*other
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = ()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
          node._backward()


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L

# See the dot graph
    
# draw_dot(L).render('L.gv', view=True)

## Now we start with the backpropagation




