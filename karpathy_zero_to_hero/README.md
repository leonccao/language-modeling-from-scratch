# micrograd practice

This is an implementation-free learning scaffold based on Andrej Karpathy's
[`micrograd`](https://github.com/karpathy/micrograd) as the first lesson of
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). It keeps
the original public interfaces and turns the scalar autograd engine, neural
network library, and training demo into exercises.

The scaffold is pinned conceptually to upstream commit
`7bc720e951fe422b8f8814aa5aa1b64121d26b4c`. There is intentionally no local
solution copy.

## Quick start

This project uses Python 3.9 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
uv run pytest --collect-only
uv run pytest
```

The tests should be discovered successfully but fail with
`NotImplementedError` at first. Implement one section at a time and rerun the
corresponding test file.

## Recommended exercise order

1. Implement `Value` initialization and representation.
2. Implement forward arithmetic operations.
3. Implement the convenience arithmetic operators.
4. Implement reverse-mode automatic differentiation.
5. Implement `Module`, `Neuron`, `Layer`, and `MLP`.
6. Complete the model, loss, optimization, and prediction exercises in
   `demo.py`.

| Stage | Command |
| --- | --- |
| Value basics | `uv run pytest tests/test_01_value_basics.py` |
| Forward operations | `uv run pytest tests/test_02_forward_ops.py` |
| Backpropagation | `uv run pytest tests/test_03_backward.py` |
| Neural networks | `uv run pytest tests/test_04_nn.py` |
| Entire exercise suite | `uv run pytest` |

## Guided demo

Install the optional scientific stack and run the guided demo:

```bash
uv sync --group demo
uv run --group demo python demo.py
```

The setup and dataset sections are complete. Later sections state their
contracts but omit model construction, loss calculation, SGD, and
decision-boundary prediction. Complete the Python package before attempting
those sections.

## Project structure

```text
micrograd/
  engine.py       # scalar Value and autograd exercises
  nn.py           # neural-network abstraction exercises
tests/            # progressive, dependency-light correctness checks
demo.py           # guided end-to-end classifier exercise
```

The tests use deterministic expected values and numerical finite differences;
PyTorch is not required. The upstream MIT license is retained in `LICENSE`.
