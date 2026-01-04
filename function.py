from functools import singledispatch, update_wrapper

def primitive(fn):
    """Create a singledispatch-backed primitive function.

    Returns a `Function` that dispatches implementations by argument
    type via `singledispatch` and exposes a `.register` method.
    """

    impl = singledispatch(fn)
    func = Function(impl)
    # forward metadata (name, docstring, etc.)
    update_wrapper(func, fn)
    # expose register
    func.register = impl.register

    return func

def as_function(obj):
    """Ensure `obj` is a `Function`.

    If `obj` is already a `Function`, return it unchanged; otherwise
    promote the value to a constant `Function`.
    """

    if isinstance(obj, Function):
        return obj
    return Function(lambda *xs: obj)

class Function:
    """
    First-class mathematical function supporting algebraic operations.
    """

    def __init__(self, impl):
        """Initialize with a callable implementation.

        The `impl` must be callable; it is stored and invoked by
        `__call__`.
        """

        if not callable(impl):
            raise TypeError("Function requires a callable")
        self._impl = impl

    def __call__(self, *xs):
        """Evaluate the wrapped function at `x`."""

        return self._impl(*xs)

    # ---------- algebra ----------

    def __add__(self, other):
        """Pointwise addition: (f + g)(x) = f(x) + g(x)."""

        other = as_function(other)
        return Function(lambda x: self(x) + other(x))

    def __radd__(self, other):
        """Right-side addition; promotes `other` and delegates to `+`."""

        return as_function(other) + self

    def __mul__(self, other):
        """Pointwise multiplication: (f * g)(x) = f(x) * g(x)."""

        other = as_function(other)
        return Function(lambda x: self(x) * other(x))

    def __rmul__(self, other):
        """Right-side multiplication; promotes `other` and delegates."""

        return as_function(other) * self

    def __neg__(self):
        """Pointwise negation: (-f)(x) = -f(x)."""

        return Function(lambda x: -self(x))

    def __sub__(self, other):
        """Pointwise subtraction: (f - g)(x) = f(x) - g(x)."""

        return self + (-other)

    def __rsub__(self, other):
        """Right-side subtraction; promote `other` and subtract `self`."""

        return as_function(other) - self

    def __truediv__(self, other):
        """Pointwise true division: (f / g)(x) = f(x) / g(x)."""

        other = as_function(other)
        return Function(lambda x: self(x) / other(x))

    def __rtruediv__(self, other):
        """Right-side true division; promote `other` and delegate."""

        return as_function(other) / self

    # ---------- composition ----------

    def compose(self, other):
        """
        (self ∘ other)(x) = self(other(x))
        """
        other = as_function(other)
        return Function(lambda x: self(other(x)))

    def __matmul__(self, other):
        """Syntactic sugar for composition using `@`."""

        return self.compose(other)

    # ---------- debugging ----------

    def __repr__(self):
        """Representation showing the wrapped implementation."""

        return f"Function({self._impl})"