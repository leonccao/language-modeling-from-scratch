class Value:
    """Store a scalar value and the gradient of an output with respect to it."""

    def __init__(self, data, _children=(), _op=""):
        """Initialize a graph node, its gradient, and its graph metadata."""
        raise NotImplementedError("TODO: implement Value.__init__")

    def __add__(self, other):
        """Return the sum as a new graph-connected Value."""
        raise NotImplementedError("TODO: implement Value.__add__")

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
        raise NotImplementedError("TODO: implement Value.backward")

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

    def __repr__(self):
        """Return the upstream-compatible data and gradient representation."""
        raise NotImplementedError("TODO: implement Value.__repr__")
