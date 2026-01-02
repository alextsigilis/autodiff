import sympy 

class Dual:
    """
    A class representing dual numbers for forward-mode automatic differentiation.
    Supports higher-order derivatives by storing dual parts as a list.
    """
    def __init__(self, real, dual=0):
        self.real = real
        self.dual = dual

    def __repr__(self):
        return f"{self.real} + {self.dual}ε"
    
    #def _repr_latex_(self):
        #return f"{self.real} + {self.dual}\\epsilon"
    
    def __add__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)
    
    def __radd__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real * other.real,
                    self.real * other.dual + self.dual * other.real)
    
    def __rmul__(self, other):      
        return self * other
    
    def __truediv__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return Dual(self.real / other.real,
                    (self.dual * other.real - self.real * other.dual) / (other.real ** 2))
    
    def __rtruediv__(self, other):
        if not isinstance(other, Dual):
            other = Dual(other)
        return other / self
    
    def __pow__(self, power):
        if isinstance(power, Dual):
            raise NotImplementedError("Raising to a dual power is not supported.")
        return Dual(self.real ** power,
                    power * self.real ** (power - 1) * self.dual)
    
    def __neg__(self):
        return Dual(-self.real, -self.dual)
    
    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.real == other.real and self.dual == other.dual
        return False
    
