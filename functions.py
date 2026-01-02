"""
Functions for automatic differentiation using dual numbers.

This module provides implementations of common mathematical functions (sin, cos, tan, exp, log)
that support forward-mode automatic differentiation via dual numbers. It uses Python's
functools.singledispatch to provide type-specific implementations for int, float, and Dual types.

For standard numeric types (int, float), the functions delegate to the math module.
For Dual inputs, the functions compute both the function value and its derivative using the chain rule,
returning a new Dual number with the appropriate real and dual parts.

Supported functions:
- sin(x): Sine function
- cos(x): Cosine function
- tan(x): Tangent function
- atan(x): Arctangent function
- atan2(y, x): Arctangent of y / x considering the quadrant
- exp(x): Exponential function
- log(x): Natural logarithm

Example:
    from dual import Dual
    from functions import sin
    x = Dual(0, 1)  # Dual number representing x with derivative 1
    result = sin(x)  # Dual(sin(0), cos(0)) = Dual(0, 1)
"""

import math
import sympy
from functools import singledispatch
from dual import Dual

# Basic function utlities 
def compose(f, g):
    """
    Returns the composition of two functions f and g.
    That is, compose(f, g)(x) = f(g(x))
    """
    def fog(*args):
        return f(g(*args))
    return fog

# sin(x) function

@singledispatch
def sin(x):
    raise Exception("sin not implemented for type", type(x))

@sin.register(int)
def _(x: int):
    return math.sin(x)

@sin.register(float)
def _(x: float):
    return math.sin(x)

@sin.register(Dual)
def _(x: Dual):
    return Dual(sin(x.real), x.dual * cos(x.real))

@sin.register(sympy.Basic)
def _(x: sympy.Basic):
    return sympy.sin(x)

@sin.register(str)
def _(x: str):
    return sympy.sin(sympy.Symbol(x))

# cos(x) function

@singledispatch
def cos(x):
    raise Exception("cos not implemented for type", type(x))

@cos.register(int)
def _(x: int):
    return math.cos(x)

@cos.register(float)
def _(x: float):
    return math.cos(x)

@cos.register(Dual)
def _(x: Dual):
    return Dual(cos(x.real), -x.dual * sin(x.real))

@cos.register(sympy.Basic)
def _(x: sympy.Basic):
    return sympy.cos(x)

@cos.register(str)
def _(x: str):
    return sympy.cos(sympy.Symbol(x))

# tan(x) function

@singledispatch
def tan(x):
    raise Exception("tan not implemented for type", type(x))

@tan.register(int)
def _(x: int):
    return math.tan(x)

@tan.register(float)
def _(x: float):
    return math.tan(x)

@tan.register(Dual)
def _(x: Dual):
    return Dual(tan(x.real), x.dual / (cos(x.real) ** 2))

@tan.register(sympy.Basic)
def _(x: sympy.Basic):
    return sympy.tan(x)

@tan.register(str)
def _(x: str):
    return sympy.tan(sympy.Symbol(x))

# atan(x) function

@singledispatch
def atan(x):
    raise Exception("atan not implemented for type", type(x))

@atan.register(int)
def _(x: int):
    return math.atan(x)

@atan.register(float)
def _(x: float):
    return math.atan(x)

@atan.register(Dual)
def _(x: Dual):
    return Dual(atan(x.real), x.dual / (1 + x.real ** 2))

@atan.register(sympy.Basic)
def _(x: sympy.Basic):
    return sympy.atan(x)

@atan.register(str)
def _(x: str):
    return sympy.atan(sympy.Symbol(x))

# atan2(y, x) function

def atan2(y, x):
    """
    Computes the arctangent of y / x considering the quadrant of the point (x, y).

    Supports int, float, Dual, sympy.Basic, and str types.

    Args:
        y: The y-coordinate
        x: The x-coordinate
    Returns:
        The angle in radians as the appropriate type
    """
    pi = math.pi if isinstance(x, (int, float)) and isinstance(y, (int, float)) else sympy.pi
    if x == 0 and y > 0:
        return pi / 2
    elif x == 0 and y < 0:
        return -pi / 2
    else:
        return atan(y / x)

# exp(x) function

@singledispatch
def exp(x):
    raise Exception("exp not implemented for type", type(x))

@exp.register(int)
def _(x: int):
    return math.exp(x)

@exp.register(float)
def _(x: float):
    return math.exp(x)

@exp.register(Dual)
def _(x: Dual):
    exp_real = exp(x.real)
    return Dual(exp_real, x.dual * exp_real)

@exp.register(sympy.Basic)
def _(x: sympy.Basic):
    return sympy.exp(x)

@exp.register(str)
def _(x: str):
    return sympy.exp(sympy.Symbol(x))

# log(x) function

@singledispatch
def log(x):
    raise Exception("log not implemented for type", type(x))

@log.register(int)
def _(x: int):
    return math.log(x)

@log.register(float)
def _(x: float):
    return math.log(x)

@log.register(Dual)
def _(x: Dual):
    return Dual(log(x.real), x.dual / x.real)

@log.register(sympy.Basic)
def _(x: sympy.Basic):
    return sympy.log(x)

@log.register(str)
def _(x: str):
    return sympy.log(sympy.Symbol(x))

# sqrt(x) function

@singledispatch
def sqrt(x):
    raise Exception("sqrt not implemented for type", type(x))

@sqrt.register(int)
def _(x: int):
    return math.sqrt(x)

@sqrt.register(float)
def _(x: float):
    return math.sqrt(x)

@sqrt.register(Dual)
def _(x: Dual):
    sqrt_real = sqrt(x.real)
    return Dual(sqrt_real, x.dual / (2 * sqrt_real))

@sqrt.register(sympy.Basic)
def _(x: sympy.Basic):
    return sympy.sqrt(x)

@sqrt.register(str)
def _(x: str):
    return sympy.sqrt(sympy.Symbol(x))

