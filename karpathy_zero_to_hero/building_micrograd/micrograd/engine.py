from collections.abc import Callable


class Value:
    """Store a scalar value and the gradient of an output with respect to it."""
    data: float
    grad: float
    _prev: set["Value"]
    _op: str
    _backward: Callable[[], None]

    def __init__(self, data, _children=(), _op=""):
        """Initialize a graph node, its gradient, and its graph metadata."""
        self.data = float(data)
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
        out._backward = _backward

        return out

    def __mul__(self, other):
        """Return the product as a new graph-connected Value."""

        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """Raise this Value to an int or float power."""
        assert not isinstance(other, Value), "Does not support Value power"
        out = Value(self.data ** other, (self,), "**")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward

        return out

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
        return self.__mul__(-1)

    def __radd__(self, other):
        """Support scalar addition with the scalar on the left."""
        return self.__add__(other)

    def __sub__(self, other):
        """Return the result of subtracting other from this Value."""
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        """Support subtraction from a scalar on the left."""
        return self.__neg__().__add__(other)

    def __rmul__(self, other):
        """Support scalar multiplication with the scalar on the left."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Return the result of dividing this Value by other."""
        return self.__mul__(other.__pow__(-1))

    def __rtruediv__(self, other):
        """Support division of a scalar by this Value."""
        return self.__pow__(-1).__mul__(other)

    def __repr__(self) -> str:
        """Return the upstream-compatible data and gradient representation."""
        return f"Value(data={self.data:.1f}, grad={self.grad:.1f})"
