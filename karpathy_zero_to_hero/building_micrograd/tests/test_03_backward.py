import pytest
from micrograd.engine import Value


def finite_difference(function, values, index, step=1e-6):
    """Estimate one partial derivative without relying on an autograd library."""
    before = list(values)
    after = list(values)
    before[index] -= step
    after[index] += step
    return (function(*after) - function(*before)) / (2 * step)


def test_backward_seeds_output_and_applies_chain_rule():
    x = Value(4.0)
    intermediate = x * 3
    output = intermediate + 2

    output.backward()

    assert output.grad == 1
    assert intermediate.grad == 1
    assert x.grad == 3


def test_backward_accumulates_gradients_across_reused_values():
    x = Value(3.0)
    output = x * x + x

    output.backward()

    assert x.grad == 7


def test_relu_backward_blocks_negative_branch():
    negative = Value(-2.0)
    positive = Value(2.0)
    output = negative.relu() + positive.relu()

    output.backward()

    assert negative.grad == 0
    assert positive.grad == 1


def test_composite_graph_matches_numerical_gradients():
    def scalar_function(x, y):
        return ((x * y + x) ** 2 + max(0, y - x)) / 3

    x = Value(2.0)
    y = Value(-3.0)
    output = ((x * y + x) ** 2 + (y - x).relu()) / 3

    output.backward()

    expected_x = finite_difference(scalar_function, [x.data, y.data], 0)
    expected_y = finite_difference(scalar_function, [x.data, y.data], 1)
    assert x.grad == pytest.approx(expected_x, rel=1e-6, abs=1e-6)
    assert y.grad == pytest.approx(expected_y, rel=1e-6, abs=1e-6)
