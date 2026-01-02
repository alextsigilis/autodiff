"""
Symbolic mathematics classes for representing math symbols and expressions as graphs.

This module provides classes to build and manipulate symbolic expressions:
- Expression: Base class for all symbolic expressions
- Symbol: Represents a symbolic variable
- Constant: Represents a numeric constant
- Function: Represents a symbolic function of multiple arguments
- Operation classes (Add, Mul, etc.): Represent binary operations
"""

class Expression:
    """
    Base class for symbolic expressions.

    Represents a node in the expression graph.
    """
    pass

class Symbol(Expression):
    """
    Represents a symbolic variable.

    Args:
        name (str): The name of the symbol (e.g., 'x', 'y')
    """
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __add__(self, other):
        return Add(self, _to_expr(other))

    def __radd__(self, other):
        return Add(_to_expr(other), self)

    def __sub__(self, other):
        return Sub(self, _to_expr(other))

    def __rsub__(self, other):
        return Sub(_to_expr(other), self)

    def __mul__(self, other):
        return Mul(self, _to_expr(other))

    def __rmul__(self, other):
        return Mul(_to_expr(other), self)

    def __truediv__(self, other):
        return Div(self, _to_expr(other))

    def __rtruediv__(self, other):
        return Div(_to_expr(other), self)

    def __pow__(self, other):
        return Pow(self, _to_expr(other))

    def __neg__(self):
        return Neg(self)

class Constant(Expression):
    """
    Represents a numeric constant.

    Args:
        value (float): The constant value
    """
    def __init__(self, value):
        self.value = float(value)

    def __repr__(self):
        return str(self.value)

    # Similar arithmetic operations as Symbol
    def __add__(self, other):
        return Add(self, _to_expr(other))

    def __radd__(self, other):
        return Add(_to_expr(other), self)

    def __sub__(self, other):
        return Sub(self, _to_expr(other))

    def __rsub__(self, other):
        return Sub(_to_expr(other), self)

    def __mul__(self, other):
        return Mul(self, _to_expr(other))

    def __rmul__(self, other):
        return Mul(_to_expr(other), self)

    def __truediv__(self, other):
        return Div(self, _to_expr(other))

    def __rtruediv__(self, other):
        return Div(_to_expr(other), self)

    def __pow__(self, other):
        return Pow(self, _to_expr(other))

    def __neg__(self):
        return Neg(self)

class Function(Expression):
    """
    Represents a symbolic function of multiple arguments.

    Args:
        name (str): The function name (e.g., 'f', 'sin')
        args (list of Expression): The arguments to the function
    """
    def __init__(self, name, *args):
        self.name = name
        self.args = [_to_expr(arg) for arg in args]

    def __repr__(self):
        arg_str = ', '.join(repr(arg) for arg in self.args)
        return f"{self.name}({arg_str})"

# Operation classes
class Add(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} + {self.right})"

class Sub(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} - {self.right})"

class Mul(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} * {self.right})"

class Div(Expression):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} / {self.right})"

class Pow(Expression):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent

    def __repr__(self):
        return f"({self.base} ** {self.exponent})"

class Neg(Expression):
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"(-{self.expr})"

# Helper function to convert values to Expression
def _to_expr(value):
    """
    Convert a value to an Expression.

    If it's already an Expression, return it.
    If it's a number, return a Constant.
    Otherwise, raise an error.
    """
    if isinstance(value, Expression):
        return value
    elif isinstance(value, (int, float)):
        return Constant(value)
    else:
        raise TypeError(f"Cannot convert {type(value)} to Expression")

# Convenience functions for common operations
def sin(expr):
    """Symbolic sine function."""
    return Function('sin', expr)

def cos(expr):
    """Symbolic cosine function."""
    return Function('cos', expr)

def exp(expr):
    """Symbolic exponential function."""
    return Function('exp', expr)

def log(expr):
    """Symbolic logarithm function."""
    return Function('log', expr)