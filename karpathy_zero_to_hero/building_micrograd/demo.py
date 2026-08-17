"""Guided end-to-end exercises for the micrograd practice project.

Complete the ``micrograd`` package first, then work through the exercise
functions in this file. The provided functions handle reproducible data setup
and plotting; the model, objective, optimization, and prediction functions are
left for you to implement.

See ``README.md`` for test and demo commands.
"""

import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
from sklearn.datasets import make_moons


def seed_random_generators(seed: int = 1337) -> None:
    """Seed Python and NumPy so data and model initialization are repeatable."""
    random.seed(seed)
    np.random.seed(seed)


def create_dataset(
    n_samples: int = 100, noise: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Return two-moons features and labels encoded as -1 and 1."""
    features, labels = make_moons(n_samples=n_samples, noise=noise)
    return features, labels * 2 - 1


def plot_dataset(features: np.ndarray, labels: np.ndarray) -> None:
    """Plot the two-dimensional training dataset."""
    plt.figure(figsize=(5, 5))
    plt.scatter(features[:, 0], features[:, 1], c=labels, s=20, cmap="jet")
    plt.title("Two-moons training data")
    plt.show()


# %% Exercise 1: construct the model
def create_model() -> MLP:
    """Create an MLP with architecture 2 -> 16 -> 16 -> 1."""
    return MLP(2, [16, 16, 1])


# %% Exercise 2: define the objective
def calculate_loss(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: Optional[int] = None,
) -> tuple[Value, float]:
    """Return regularized max-margin loss and classification accuracy.

    Optionally select a random batch, evaluate the model, and combine the mean
    binary max-margin loss with L2 parameter regularization. The returned loss
    must be a Value and the returned accuracy must be a Python number.
    """
    inputs = [list(map(Value, row)) for row in features]
    scores = list(map(model, inputs))

    losses = [(1 - yi * scorei).relu() for yi, scorei in zip(labels, scores)]
    avg_loss = sum(losses) * (1.0 / len(losses))

    alpha = 1e-4
    reg_loss = alpha * sum(p * p for p in model.parameters())
    total_loss = avg_loss + reg_loss

    accs = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(labels, scores)]
    avg_acc = sum(accs) * (1.0 / len(accs))

    return (total_loss, avg_acc)


# %% Exercise 3: optimize with SGD
def train(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    steps: int = 100,
) -> None:
    """Train the model with SGD and a linearly decreasing learning rate.

    Each step must evaluate the objective, clear existing gradients,
    backpropagate, update every parameter, and report loss and accuracy. Use a
    learning rate that decreases from 1.0 toward 0.1 over the training run.
    """

    for step in range(steps):
        loss, acc = calculate_loss(model, features, labels)

        model.zero_grad()
        loss.backward()

        learn_rate = 1.0 - (1.0 - 0.1) / 100 * step
        for p in model.parameters():
            p.data -= p.grad * learn_rate

        print(f"Step {step}, loss {loss.data}, accuracy {acc * 100}%")


def create_decision_mesh(
    features: np.ndarray, step: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinate grids and flattened points covering the dataset."""
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step),
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, mesh_points


# %% Exercise 4: predict the decision boundary
def predict_mesh(
    model: MLP, mesh_points: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    """Return boolean model predictions reshaped to output_shape."""
    inputs = [list(map(Value, xrow)) for xrow in mesh_points]
    scores = list(map(model, inputs))
    result = np.array([score.data > 0 for score in scores])
    result = result.reshape(output_shape)
    return result


def plot_decision_boundary(
    features: np.ndarray,
    labels: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Plot mesh predictions behind the original training samples."""
    plt.figure()
    plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(
        features[:, 0],
        features[:, 1],
        c=labels,
        s=40,
        cmap=plt.cm.Spectral,
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Learned decision boundary")
    plt.show()


def main() -> None:
    """Run the complete data, training, and visualization workflow."""
    seed_random_generators()
    features, labels = create_dataset()
    plot_dataset(features, labels)

    model = create_model()
    print(model)
    print("number of parameters", len(model.parameters()))

    initial_loss, initial_accuracy = calculate_loss(model, features, labels)
    print(initial_loss, initial_accuracy)

    train(model, features, labels)

    xx, yy, mesh_points = create_decision_mesh(features)
    predictions = predict_mesh(model, mesh_points, xx.shape)
    plot_decision_boundary(features, labels, xx, yy, predictions)


if __name__ == "__main__":
    main()
