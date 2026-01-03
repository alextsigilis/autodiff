"""
    mathfunctions.py

Defines the math function avaiable in autodiff
"""
import math
from function import primitive

#
# Function Definitions
# --------------------

@primitive
def sqrt(x):
    raise NotImplementedError(f'sqrt not yet implemented for type {type(x)}')

# Trig Functions

@primitive
def sin(x):
    raise NotImplementedError(f'sin not yet implemented for type {type(x)}')

@primitive
def cos(x):
    raise NotImplementedError(f'cos not yet implemented for type {type(x)}')

@primitive
def tan(x):
    raise NotImplementedError(f'tan not yet implemented for type {type(x)}')

# Inverse trig functions

@primitive
def asin(x):
    raise NotImplementedError(f'asin not yet implemented for type {type(x)}')

@primitive
def acos(x):
    raise NotImplementedError(f'acos not yet implemented for type {type(x)}')

@primitive
def atan(x):
    raise NotImplementedError(f'atan not yet implemented for type {type(x)}')

# Exponetial and Logarithms

@primitive
def exp(x):
    raise NotImplementedError(f'exp not yet implemented for type {type(x)}')

@primitive
def log(x):
    raise NotImplementedError(f'log not yet implemented for type {type(x)}')

@primitive
def log10(x):
    raise NotImplementedError(f'log 10 not yet implemented for type {type(x)}')

@primitive
def log2(x):
    raise NotImplementedError(f'log 10 not yet implemented for type {type(x)}')

#
# Implementation for `float` and `int`
# -----------------------------------

@sqrt.register(int)
@sqrt.register(float)
def _(x):
    return math.sqrt(x)

# Trig implementations

@sin.register(int)
@sin.register(float)
def _(x):
    return math.sin(x)

@cos.register(int)
@cos.register(float)
def _(x):
    return math.cos(x)

@tan.register(int)
@tan.register(float)
def _(x):
    return math.tan(x)

# Inverse trig implementations

@asin.register(int)
@asin.register(float)
def _(x):
    return math.asin(x)

@acos.register(int)
@acos.register(float)
def _(x):
    return math.acos(x)

@atan.register(int)
@atan.register(float)
def _(x):
    return math.atan(x)

# Exponential / logarithm implementations

@exp.register(int)
@exp.register(float)
def _(x):
    return math.exp(x)

@log.register(int)
@log.register(float)
def _(x):
    return math.log(x)

@log10.register(int)
@log10.register(float)
def _(x):
    return math.log10(x)

@log2.register(int)
@log2.register(float)
def _(x):
    return math.log2(x)

