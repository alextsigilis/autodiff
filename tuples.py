"""
Custom tuple classes for multi-dimensional data handling.

This module defines Tuple, Up, and Down classes that provide enhanced tuple-like
functionality, including multi-dimensional indexing and custom string representations.
These classes are useful for handling structured data in automatic differentiation
and symbolic computation contexts.
"""
from functools import reduce
from operator import add
from collections.abc import Iterable

def isscalar(x) -> bool:
    return not isinstance(x, Tuple)

# Base Tuple class

class Tuple(Iterable):
    """
    A custom tuple-like class that supports multi-dimensional indexing.

    This class wraps a tuple and provides additional functionality for
    nested indexing using tuples as indices.
    """
    def __init__(self, items):
        self.items = tuple(items)

    def __iter__(self):
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            current = self
            for i in index:
                current = current.items[i]
            return current
        else:
            return self.items[index]
        
    def __repr__(self):
        item_str = ', '.join(repr(item) for item in self.items)
        return f"<{item_str}>"
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.items == other.items
    
    def __add__(self, other):
        if isinstance(other, type(self)) and len(self) == len(other):
            T = type(self)
            return T([a + b for a, b in zip(self.items, other.items)])
        return NotImplemented
    
    def __radd__(self, other):
        if isinstance(other, type(self)) and len(self) == len(other):
            T = type(self)
            return T([a + b for a, b in zip(other.items, self.items)])
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, type(self)) and len(self) == len(other):
            T = type(self)
            return T([a - b for a, b in zip(self.items, other.items)])
        return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, type(self)) and len(self) == len(other):
            T = type(self)
            return T([a - b for a, b in zip(other.items, self.items)])
        return NotImplemented
    
    def __neg__(self):
        T = type(self)
        return T([-item for item in self.items])

    def __mul__(self, other):
        if isscalar(other):
            T = type(self)
            return T([item * other for item in self.items])
        elif compatible_for_contraction(self, other):
            return reduce(add, (a * b for a, b in zip(self, other)))
        else:
            T = type(other)
            return T(*(self * b for b in other))
        
    def __rmul__(self, other):
        if isscalar(other):
            T = type(self)
            return T([other * item for item in self.items])
        elif compatible_for_contraction(other, self):
            return reduce(add, (a * b for a, b in zip(other, self)))
        else:
            T = type(other)
            return T([other * a for a in self])
        
    def map(self, func):
        """
        Apply a function to each element of the Tuple.

        Args:
            func: A function that takes a single argument.

        Returns:
            A new Tuple with the function applied to each element.
        """
        T = type(self)
        return T([func(item) for item in self.items])
    
    def recursive_map(self, func):
        """
        Recursively apply a function to each element of the Tuple.

        If an element is itself a Tuple, the function is applied recursively.

        Args:
            func: A function that takes a single argument.

        Returns:
            A new Tuple with the function applied recursively to each element.
        """
        T = type(self)
        def apply_func(item):
            if isinstance(item, Tuple):
                return item.recursive_map(func)
            else:
                return func(item)
        return T([apply_func(item) for item in self.items])
    


def compatible_for_contraction(a, b) -> bool:
    """
    Check if two Tuple instances are compatible for contraction.

    Two Tuples are compatible for contraction if
    1. They are opposite types (one Up, one Down)
    2. They have the same length
    3. Their corresponding elements are compatible for contraction.

    Args:
        a : The first Tuple instance.
        b : The second Tuple instance.

    Returns:
        bool: True if compatible for contraction, False otherwise.
    """
    if not isinstance(a, Tuple) or not isinstance(b, Tuple):
        return False
    if type(a) == type(b):
        return False
    if len(a) != len(b):
        return False
    return True


def latex_up(up_tuple):
    """
    Generate LaTeX representation of an Up tuple as a column pmatrix.
    
    Handles nested Up tuples recursively. Elements can be floats, ints, Duals, or sympy symbols.
    
    Args:
        up_tuple: An instance of Up tuple.
    
    Returns:
        str: LaTeX string representing the Up tuple as a pmatrix.
    """
    if not isinstance(up_tuple, Up):
        raise ValueError("Input must be an Up tuple")
    
    def element_latex(elem):
        if isinstance(elem, Up):
            return latex_up(elem)
        elif hasattr(elem, '_repr_latex_'):
            latex = elem._repr_latex_()
            if latex.startswith('$') and latex.endswith('$'):
                return latex[1:-1]
            else:
                return latex
        else:
            return str(elem)
    
    elements = [element_latex(item) for item in up_tuple]
    return r"\begin{pmatrix}" + r" \\ \\ ".join(elements) + r"\end{pmatrix}"


def latex_down(down_tuple):
    """
    Generate LaTeX representation of a Down tuple as a row bmatrix.
    
    Handles nested Down tuples recursively. Elements can be floats, ints, Duals, or sympy symbols.
    
    Args:
        down_tuple: An instance of Down tuple.
    
    Returns:
        str: LaTeX string representing the Down tuple as a bmatrix.
    """
    if not isinstance(down_tuple, Down):
        raise ValueError("Input must be a Down tuple")
    
    def element_latex(elem):
        if isinstance(elem, Down):
            return latex_down(elem)
        elif hasattr(elem, '_repr_latex_'):
            latex = elem._repr_latex_()
            if latex.startswith('$') and latex.endswith('$'):
                return latex[1:-1]
            else:
                return latex
        else:
            return str(elem)
    
    elements = [element_latex(item) for item in down_tuple]
    return r"\begin{bmatrix}" + r" & ".join(elements) + r"\end{bmatrix}"


def opposite_tuple_type(t):
    """
    Get the opposite Tuple type.

    Args:
        t: An instance of Up or Down tuple.

    Returns:
        The opposite Tuple class (Up if input is Down, Down if input is Up).
    """
    if isinstance(t, Up):
        return Down
    elif isinstance(t, Down):
        return Up
    else:
        return type(t)


# Up Tuple class

class Up(Tuple):
    """
    A variant of Tuple that uses parentheses in its string representation.

    Inherits all functionality from Tuple.
    """
    def __init__(self, *items):
        super().__init__(*items)
    
    def __repr__(self):
        item_str = ', '.join(repr(item) for item in self.items)
        return f"({item_str})"
    
    def _repr_latex_(self):
        s = latex_up(self)
        return f"${s}$"   

# # Function constructor
def up(*args):
    return Up(args)

# Down Tuple class

class Down(Tuple):
    """
    A variant of Tuple that uses square brackets in its string representation.

    Inherits all functionality from Tuple.
    """
    def __init__(self, *items):
        super().__init__(*items)
    
    def __repr__(self):
        item_str = ', '.join(repr(item) for item in self.items)
        return f"[{item_str}]"
    
    def _repr_latex_(self):
        s = latex_down(self)
        return f"${s}$"
    
# # Function Constructor
def down(*args):
    return Down(args)