import random

from micrograd.engine import Value


class Module:
    """Base class for objects that own trainable scalar parameters."""

    def zero_grad(self):
        """Reset every parameter gradient owned by this module to zero."""
        raise NotImplementedError("TODO: implement Module.zero_grad")

    def parameters(self):
        """Return this module's trainable Value objects."""
        raise NotImplementedError("TODO: implement Module.parameters")


class Neuron(Module):
    """A scalar-output neuron with an optional ReLU activation."""

    def __init__(self, nin, nonlin=True):
        """Create a neuron with nin inputs and trainable weights and bias."""
        raise NotImplementedError("TODO: implement Neuron.__init__")

    def __call__(self, x):
        """Evaluate the neuron for one sequence of input scalars or Values."""
        raise NotImplementedError("TODO: implement Neuron.__call__")

    def parameters(self):
        """Return the neuron's weights followed by its bias."""
        raise NotImplementedError("TODO: implement Neuron.parameters")

    def __repr__(self):
        """Describe the neuron by activation type and input width."""
        raise NotImplementedError("TODO: implement Neuron.__repr__")


class Layer(Module):
    """A group of neurons sharing the same input width."""

    def __init__(self, nin, nout, **kwargs):
        """Create nout neurons, forwarding keyword options to each neuron."""
        raise NotImplementedError("TODO: implement Layer.__init__")

    def __call__(self, x):
        """Return one Value for a width-one layer, otherwise a list of Values."""
        raise NotImplementedError("TODO: implement Layer.__call__")

    def parameters(self):
        """Return all neuron parameters as one flat list."""
        raise NotImplementedError("TODO: implement Layer.parameters")

    def __repr__(self):
        """Describe the neurons contained in this layer."""
        raise NotImplementedError("TODO: implement Layer.__repr__")


class MLP(Module):
    """A sequence of fully connected layers with a linear final layer."""

    def __init__(self, nin, nouts):
        """Build layers from input width nin through each width in nouts."""
        raise NotImplementedError("TODO: implement MLP.__init__")

    def __call__(self, x):
        """Evaluate the input through each layer in order."""
        raise NotImplementedError("TODO: implement MLP.__call__")

    def parameters(self):
        """Return every layer parameter as one flat list."""
        raise NotImplementedError("TODO: implement MLP.parameters")

    def __repr__(self):
        """Describe the layers contained in this MLP."""
        raise NotImplementedError("TODO: implement MLP.__repr__")
