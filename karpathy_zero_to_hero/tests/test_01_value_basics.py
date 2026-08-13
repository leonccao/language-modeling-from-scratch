from micrograd.engine import Value


def test_value_initializes_scalar_and_graph_state():
    value = Value(2.5)

    assert value.data == 2.5
    assert value.grad == 0
    assert value._prev == set()
    assert value._op == ""
    assert callable(value._backward)


def test_value_records_children_and_operation():
    left = Value(1.0)
    right = Value(2.0)
    value = Value(3.0, (left, right), "+")

    assert value._prev == {left, right}
    assert value._op == "+"


def test_value_repr_matches_public_contract():
    value = Value(-3.0)
    value.grad = 1.5

    assert repr(value) == "Value(data=-3.0, grad=1.5)"
