import random

from micrograd.engine import Value
from micrograd.nn import MLP, Layer, Module, Neuron


def test_base_module_has_no_parameters_and_can_zero_grad():
    module = Module()

    assert module.parameters() == []
    assert module.zero_grad() is None


def test_neuron_initializes_expected_parameters():
    random.seed(1337)
    neuron = Neuron(3, nonlin=False)

    assert len(neuron.w) == 3
    assert all(isinstance(weight, Value) for weight in neuron.w)
    assert all(-1 <= weight.data <= 1 for weight in neuron.w)
    assert isinstance(neuron.b, Value)
    assert neuron.b.data == 0
    assert neuron.nonlin is False
    assert neuron.parameters() == neuron.w + [neuron.b]


def test_neuron_forward_can_be_linear_or_relu():
    linear = Neuron(2, nonlin=False)
    linear.w[0].data = 2.0
    linear.w[1].data = -3.0
    linear.b.data = 1.0

    relu = Neuron(2, nonlin=True)
    relu.w[0].data = 2.0
    relu.w[1].data = -3.0
    relu.b.data = 1.0

    assert linear([4.0, 5.0]).data == -6.0
    assert relu([4.0, 5.0]).data == 0


def test_layer_output_shape_and_parameter_flattening():
    wide = Layer(3, 2)
    narrow = Layer(3, 1)

    wide_output = wide([1.0, 2.0, 3.0])
    narrow_output = narrow([1.0, 2.0, 3.0])

    assert isinstance(wide_output, list)
    assert len(wide_output) == 2
    assert isinstance(narrow_output, Value)
    assert len(wide.parameters()) == 8
    assert len(narrow.parameters()) == 4


def test_mlp_builds_linear_final_layer_and_flattens_parameters():
    model = MLP(2, [3, 1])

    assert len(model.layers) == 2
    assert all(neuron.nonlin for neuron in model.layers[0].neurons)
    assert all(not neuron.nonlin for neuron in model.layers[-1].neurons)
    assert len(model.parameters()) == 13
    assert isinstance(model([1.0, -1.0]), Value)


def test_zero_grad_resets_every_model_parameter():
    model = MLP(2, [3, 1])
    for parameter in model.parameters():
        parameter.grad = 7

    model.zero_grad()

    assert all(parameter.grad == 0 for parameter in model.parameters())


def test_one_sgd_step_can_reduce_squared_error():
    model = MLP(1, [1])
    neuron = model.layers[0].neurons[0]
    neuron.w[0].data = 0.5
    neuron.b.data = 0.0

    prediction = model([2.0])
    loss = prediction**2
    loss.backward()
    before = loss.data

    for parameter in model.parameters():
        parameter.data -= 0.1 * parameter.grad

    after = model([2.0]) ** 2
    assert after.data < before
