from functools import singledispatch
from collections.abc import Iterable
from dual import Dual
from function import Function

# x -> Dual(x)

@singledispatch
def to_dual(x, dual=1):
    return Dual(x, dual)

@to_dual.register(tuple)
def _(xs, dual=1):
    return tuple(to_dual(x, dual) for x in xs)

# Dual(x) -> x

@singledispatch
def from_dual(y):
    return 0

@from_dual.register(Dual)
def _(y):
    return y.dual

@from_dual.register(tuple)
def _(ys):
    return tuple(from_dual(y) for y in ys)

# Partial Derivatives

def compute_partial_at(i, f, *args):
    args = list(args)
    args[i] = to_dual(args[i])
    res = f(*args)
    return from_dual(res)

def partial(i, f = None):
    if not f:
        return lambda f: Function(lambda *args: compute_partial_at(i, f, *args))
    else:
        return Function(lambda *args: compute_partial_at(i, f, *args))
    
# Class For full derivative



class DiffOp:
    def __ini__(self, order = 1):
        if order != 1: raise NotImplementedError()

    def __call__(cls, f: Function) -> Function:
        def derivative(*args):
            if len(args) == 0: raise Exception('WTF?')
            if len(args) == 1: return compute_partial_at(0, f, *args)
            return [compute_partial_at(i, f, *args) for i in range(len(args))]
        return derivative
