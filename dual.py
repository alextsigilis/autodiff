"""Dual numbers for forward-mode automatic differentiation.

This module provides a small `Dual` class representing numbers of the
form `a + bε` (with ε^2 = 0) which is useful for computing
first-order derivatives via forward-mode automatic differentiation.

The arithmetic operators implement the usual rules for dual numbers
so that the dual part carries derivative information.
"""

from __future__ import annotations
from typing import Protocol
from mathfunctions import *

class Scalar(Protocol):
    """Protocol for a scalar value."""
    def __neg__(self) -> Scalar:
        ...
    def __add__(self, other) -> Scalar:
        ...
    def __mul__(self, other) -> Scalar:
        ...
    def __rmul__(self, other) -> Scalar:
        ...
    def __sub__(self, other) -> Scalar:
        ...
    def __rsub__(self, other) -> Scalar:
        ...
    def __truediv__(self, other) -> Scalar:
        ...
    def __rtruediv__(self, other) -> Scalar:
        ...
    def __pow__(self, other) -> Scalar:
        ...
    def __eq__(self, other) -> bool:
        ...

class Dual:
    """
    A class representing dual numbers for forward-mode automatic differentiation.
    Supports higher-order derivatives by storing dual parts as a list.
    """
    def __init__(self,
                 real: Scalar,
                 dual: Scalar | None = None):
        self.real = real
        self.dual = dual if dual else 0*real

    def __repr__(self) -> str:
        return f"{self.real} + ({self.dual})ε"
 
    def __add__(self, other: Scalar) -> Dual:
        y = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real + y.real, self.dual + y.dual)

    def __radd__(self, other: Scalar) -> Dual:
        y = other if isinstance(other, Dual) else Dual(other)
        return self + y

    def __sub__(self, other: Scalar) -> Dual:
        y = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real - y.real, self.dual - y.dual)

    def __rsub__(self, other: Scalar) -> Dual:
        y = other if isinstance(other, Dual) else Dual(other)
        return other - y

    def __mul__(self, other: Scalar) -> Dual:
        y = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real * y.real,
                    self.real * y.dual + self.dual * y.real)
    
    def __rmul__(self, other: Scalar) -> Dual:     
        return self * other
    
    def __truediv__(self, other: Scalar) -> Dual:
        y = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real / y.real,
                    (self.dual * y.real - self.real * y.dual) / (y.real ** 2))
    
    def __rtruediv__(self, other: Scalar) -> Dual:
        y = other if isinstance(other, Dual) else Dual(other)
        return y / self
    
    def __pow__(self, power: Scalar) -> Dual:
        if isinstance(power, Dual):
            raise NotImplementedError("Raising to a dual power is not supported.")
        return Dual(self.real ** power,
                    power * self.real ** (power - 1) * self.dual)
    
    def __neg__(self) -> Dual:
        return Dual(-self.real, -self.dual)
    
#
# Function Implementations
# ------------------------

@sqrt.register(Dual)
def _(x):
    return Dual(sqrt(x.real), 
                x.dual / (2 * sqrt(x.real)))

# Trig implementations

@sin.register(Dual)
def _(x):
    return Dual(sin(x.real), cos(x.real) * x.dual)

@cos.register(Dual)
def _(x):
    return Dual(cos(x.real), -sin(x.real) * x.dual)

@tan.register(Dual)
def _(x):
    return Dual(tan(x.real), 
                (1 / cos(x.real) * (1 / cos(x.real) * x.dual)))

# Inverse trig implementations

@asin.register(Dual)
def _(x):
    return Dual(asin(x.real), x.dual / (sqrt(1 - x.real ** 2)))


@acos.register(Dual)
def _(x):
    return Dual(acos(x.real), -x.dual / (sqrt(1 - x.real ** 2)))


@atan.register(Dual)
def _(x):
    return Dual(atan(x.real), x.dual / (1 + x.real ** 2))

# Exponential / logarithm implementations

@exp.register(Dual)
def _(x):
    val = exp(x.real)
    return Dual(val, val * x.dual)


@log.register(Dual)
def _(x):
    return Dual(log(x.real), x.dual / x.real)


@log10.register(Dual)
def _(x):
    return Dual(log10(x.real), x.dual / (x.real * log(10)))


@log2.register(Dual)
def _(x):
    return Dual(log2(x.real), x.dual / (x.real * log(2)))

