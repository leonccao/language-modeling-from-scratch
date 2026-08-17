import random

from micrograd.engine import Value


class Module:
    """Base class for objects that own trainable scalar parameters."""

    def zero_grad(self):
        """Reset every parameter gradient owned by this module to zero."""
        for para in self.parameters():
            para.grad = 0

    def parameters(self):
        """Return this module's trainable Value objects."""
        return []


class Neuron(Module):
    """A scalar-output neuron with an optional ReLU activation."""

    w: list[float]
    b: list[float]
    nonlin: bool

    def __init__(self, nin, nonlin=True):
        """Create a neuron with nin inputs and trainable weights and bias."""
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """Evaluate the neuron for one sequence of input scalars or Values."""
        res = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return res.relu() if self.nonlin else res

    def parameters(self):
        """Return the neuron's weights followed by its bias."""
        return self.w + [self.b]

    def __repr__(self):
        """Describe the neuron by activation type and input width."""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neruon({len(self.w)})"


class Layer(Module):
    """A group of neurons sharing the same input width."""

    neurons: list[Neuron]

    def __init__(self, nin, nout, **kwargs):
        """Create nout neurons, forwarding keyword options to each neuron."""
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """Return one Value for a width-one layer, otherwise a list of Values."""
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """Return all neuron parameters as one flat list."""
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """Describe the neurons contained in this layer."""
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A sequence of fully connected layers with a linear final layer."""

    layers: list[Layer]

    def __init__(self, nin, nouts):
        """Build layers from input width nin through each width in nouts."""
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(sz) - 2) for i in range(len(sz) - 1)
        ]

    def __call__(self, x):
        """Evaluate the input through each layer in order."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Return every layer parameter as one flat list."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """Describe the layers contained in this MLP."""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
