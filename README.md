# Automatic Differentiation Library

A Python library for automatic differentiation using dual numbers and custom tuple structures.

## Features

- **Dual Numbers**: Forward-mode automatic differentiation with support for higher-order derivatives
- **Tuple Structures**: Custom `Up` and `Down` tuple classes for multi-dimensional data handling
- **Differentiation Operators**: Callable differentiation operator `D` with composition and power operations
- **LaTeX Support**: Built-in LaTeX representations for mathematical expressions
- **Symbolic Integration**: Compatible with SymPy for symbolic computations

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd autodiff
   ```

2. Install dependencies (if any):
   ```bash
   pip install sympy
   ```

## Usage

### Basic Differentiation

```python
from dual import Dual
from derivatives import D

# Create a dual number
x = Dual(2, 1)  # 2 + 1ε
f = x ** 2 + 3 * x + 1
print(f)  # 2 + 1ε → derivative is 1

# Use differentiation operator
def func(x):
    return x ** 2

df = D(func)
print(df(2))  # Computes gradient at x=2
```

### Tuple Operations

```python
from tuples import Up, Down

# Create up and down tuples
u = Up(1, 2, 3)
d = Down(4, 5, 6)

print(u + u)  # Element-wise addition
print(d * d)  # Contraction if compatible
```

### Higher-Order Derivatives

```python
# Second derivative
d2f = D ** 2
print(d2f(func)(2))
```

## Project Structure

- `dual.py`: Dual number implementation
- `tuples.py`: Custom tuple classes with LaTeX support
- `derivatives.py`: Differentiation operators and utilities
- `functions.py`: Additional mathematical functions
- `examples.ipynb`: Jupyter notebook with usage examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.