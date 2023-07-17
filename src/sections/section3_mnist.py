"""
MNIST dataset
"""
from __future__ import annotations

from src.support.mnist import load_mnist


def call_mnist() -> None:
    """
    load_mnist
    """
    (training_images, training_labels), (test_images, test_labels) = load_mnist(normalize=False)
    print(training_images.shape)
    print(training_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)
