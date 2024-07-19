## Mircro grad from scratch

class Value: #Because we want a graph

    def __init__(self,data,_children = (), _op =''): # Underscore means the variable is private, read the readme for more

        self.data = data
        self._prev = set(_children)

    def __repr__(self): # Double 
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        out = Value(self.data + other.data, (self,other),'+')
        return out

    def __mul__(self, other):
        out = Value(self.data*other.data,(self,other),'*')
        return out
    
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b +c
d
