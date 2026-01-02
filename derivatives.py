"""
Module for computing derivatives using automatic differentiation with dual numbers.

This module provides functions to compute partial derivatives and gradients
of functions using forward-mode automatic differentiation implemented with
dual numbers and tuple structures.
"""

from typing import Callable
from tuples import Tuple, Up, Down, isscalar
from dual import Dual

def partial_at(f: Callable, i: int, *args):
    """
    Compute the partial derivative of a function f with respect to its i-th argument at the given point.
    
    Uses forward-mode automatic differentiation by setting the dual part of the i-th argument to 1.
    
    Args:
        f (Callable): The function to differentiate. Should accept the same number of arguments as provided in *args.
        i (int): The index of the argument to differentiate with respect to (0-based).
        *args: The point at which to evaluate the derivative. Can be scalars or tuple structures.
    
    Returns:
        The partial derivative value at the given point. Type matches the output of f.
    """
    x = Up(Dual(args[j],1) if j == i else Dual(args[j],0)
             for j in range(len(args)))
    y = f(*x)
    return y.dual if isscalar(y) else y.recursive_map(lambda z: z.dual)


def partial(f: Callable, i: int):
    """
    Return a function that computes the partial derivative of f with respect to its i-th argument.
    
    Args:
        f (Callable): The function to differentiate.
        i (int): The index of the argument to differentiate with respect to.
    
    Returns:
        Callable: A function that takes *args and returns the partial derivative of f at that point.
    """
    df = lambda *args: partial_at(f, i, *args)
    return df

def derivative_at(f: Callable, *args):
    """
    Compute the gradient (vector of partial derivatives) of a function f at the given point.
    
    Args:
        f (Callable): The function to differentiate.
        *args: The point at which to evaluate the gradient.
    
    Returns:
        Down: A Down tuple containing the partial derivatives with respect to each argument.
    """
    args = Up(args)
    gradient = [partial_at(f, i, *args) for i in range(len(args))]
    return Down(gradient) if len(gradient) > 1 else Down(gradient[0])


def derivative(f: Callable):
    """
    Return a function that computes the gradient of f.
    
    Args:
        f (Callable): The function to differentiate.
    
    Returns:
        Callable: A function that takes *args and returns the gradient of f at that point as a Down tuple.
    """
    def df(*args):
        return derivative_at(f, *args)
    return df


class DiffOp:
    """
    Differentiation operator class.
    
    Represents the derivative operator of a given order. Can be applied to functions,
    multiplied for composition, and raised to powers for higher-order derivatives.
    """
    
    def __init__(self, order=1):
        self.order = order
    
    def __call__(self, f):
        """
        Apply the differentiation operator to a function.
        
        Args:
            f (Callable): The function to differentiate.
        
        Returns:
            Callable: The differentiated function.
        """
        df = f
        for _ in range(self.order):
            df = derivative(df)
        return df
    
    def __mul__(self, other):
        """
        Compose differentiation operators.
        
        Args:
            other (DiffOp): Another differentiation operator.
        
        Returns:
            DiffOp: A new operator representing the composition.
        """
        if isinstance(other, DiffOp):
            return DiffOp(self.order + other.order)
        return NotImplemented
    
    def __pow__(self, n):
        """
        Raise the differentiation operator to a power.
        
        Args:
            n (int): The power to raise to.
        
        Returns:
            DiffOp: A new operator representing the nth derivative.
        """
        return DiffOp(self.order * n)
    
    def __repr__(self):
        if self.order == 1:
            return "D"
        else:
            return f"D^{self.order}"


# Singleton instance of the differentiation operator
D = DiffOp(order=1)