from collections.abc import Callable


class Value:
    """Store a scalar value and the gradient of an output with respect to it."""
    data: float
    grad: float
    _prev: set["Value"]
    _op: str
    _backward: Callable[[], None]

    """
    TODO
    - backward
    - a * 2 -> 2 * a
    """

    def __init__(self, data, _children=(), _op=""):
        """Initialize a graph node, its gradient, and its graph metadata."""
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None


    def __add__(self, other) -> "Value":
        """Return the sum as a new graph-connected Value."""

        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = _backward

        return out

    def __mul__(self, other):
        """Return the product as a new graph-connected Value."""
        raise NotImplementedError("TODO: implement Value.__mul__")

    def __pow__(self, other):
        """Raise this Value to an int or float power."""
        raise NotImplementedError("TODO: implement Value.__pow__")

    def relu(self):
        """Apply the scalar rectified linear unit operation."""
        raise NotImplementedError("TODO: implement Value.relu")

    def backward(self):
        """Populate gradients for the computation graph ending at this Value."""
        """
        TODO
        topo sort
        """
        self._backward()

    def __neg__(self):
        """Return the arithmetic negation of this Value."""
        raise NotImplementedError("TODO: implement Value.__neg__")

    def __radd__(self, other):
        """Support scalar addition with the scalar on the left."""
        raise NotImplementedError("TODO: implement Value.__radd__")

    def __sub__(self, other):
        """Return the result of subtracting other from this Value."""
        raise NotImplementedError("TODO: implement Value.__sub__")

    def __rsub__(self, other):
        """Support subtraction from a scalar on the left."""
        raise NotImplementedError("TODO: implement Value.__rsub__")

    def __rmul__(self, other):
        """Support scalar multiplication with the scalar on the left."""
        raise NotImplementedError("TODO: implement Value.__rmul__")

    def __truediv__(self, other):
        """Return the result of dividing this Value by other."""
        raise NotImplementedError("TODO: implement Value.__truediv__")

    def __rtruediv__(self, other):
        """Support division of a scalar by this Value."""
        raise NotImplementedError("TODO: implement Value.__rtruediv__")

    def __repr__(self) -> str:
        """Return the upstream-compatible data and gradient representation."""
        return f"Value(data={self.data:.1f}, grad={self.grad:.1f})"
