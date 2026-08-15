import pytest
from micrograd.engine import Value


def test_addition_accepts_values_and_scalars():
    left = Value(2.0)
    right = Value(-3.0)

    assert (left + right).data == -1.0
    assert (left + 4).data == 6.0
    assert (4 + left).data == 6.0


def test_multiplication_accepts_values_and_scalars():
    left = Value(2.0)
    right = Value(-3.0)

    assert (left * right).data == -6.0
    assert (left * 4).data == 8.0
    assert (4 * left).data == 8.0


def test_power_accepts_numeric_exponents_only():
    value = Value(3.0)

    assert (value**2).data == 9.0
    assert (value**-1).data == pytest.approx(1 / 3)
    with pytest.raises(AssertionError):
        value ** Value(2.0)


def test_relu_handles_negative_zero_and_positive_values():
    assert Value(-2.0).relu().data == 0
    assert Value(0.0).relu().data == 0
    assert Value(2.0).relu().data == 2.0


def test_convenience_operators_match_scalar_arithmetic():
    value = Value(4.0)

    assert (-value).data == -4.0
    assert (value - 3).data == 1.0
    assert (10 - value).data == 6.0
    assert (value / 2).data == 2.0
    assert (8 / value).data == 2.0
