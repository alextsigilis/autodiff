"""Dual numbers for forward-mode automatic differentiation.

This module provides a small `Dual` class representing numbers of the
form `a + bε` (with ε^2 = 0) which is useful for computing
first-order derivatives via forward-mode automatic differentiation.

The arithmetic operators implement the usual rules for dual numbers
so that the dual part carries derivative information.
"""

from __future__ import annotations
import mathfunctions as math
from typing import Protocol

class Scalar(Protocol):
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
        return f"{self.real} + {self.dual}ε"
 
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
 
    
@math.sin.register(Dual)
def _(x: Dual) -> Dual:
    return Dual(math.sin(x.real), math.cos(x.real) * x.dual)

@math.cos.register(Dual)
def _(x: Dual) -> Dual:
    return Dual(math.cos(x.real), -math.sin(x.real) * x.dual)