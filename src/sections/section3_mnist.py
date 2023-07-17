"""
MNIST dataset
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from src.modules.lower_layer_modules.Numbers_numpy import softmax, sigmoid
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

    print(training_labels[0])
    image_array: np.ndarray = training_images[0]
    print(image_array.shape)
    image_array = image_array.reshape(28, 28)
    print(image_array.shape)
    show_image(image_array)


def make_sample_prediction(sample_weight_pickle_file: Path) -> None:
    """
    make_prediction
    """
    training_images, training_labels = get_data()
    network: np.ndarray = init_network(sample_weight_pickle_file)
    batch_size: int = 100
    accuracy_count: int = 0
    for image_index in range(0, len(training_images), batch_size):
        input_batch: np.ndarray = training_images[image_index:image_index + batch_size]
        prediction_batch: np.ndarray = predict(network, input_batch)
        prediction_index: int = cast(int, np.argmax(prediction_batch, axis=1))
        accuracy_count += np.sum(prediction_index == training_labels[image_index:image_index + batch_size])
    print("Accuracy:" + str(float(accuracy_count) / len(training_images)))


def show_image(image_array: np.ndarray) -> None:
    """
    show_image
    """
    plt.imshow(image_array, cmap="gray")
    plt.show()
    plt.close()


def get_data() -> tuple[np.ndarray, np.ndarray]:
    """
    get_Data
    """
    (training_images, training_labels), (_test_images, _test_labels) = load_mnist(normalize=True)
    return training_images, training_labels


def init_network(sample_weight_pickle_file: Path) -> np.ndarray:
    """
    init_network
    """
    with open(sample_weight_pickle_file, "rb") as f:
        network: np.ndarray = pickle.load(f)
    return network


def predict(network: np.ndarray, input_array: np.ndarray) -> np.float64:
    """
    predict
    """
    weight1, weight2, weight3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1: np.ndarray = np.dot(input_array, weight1) + b1
    z1: np.ndarray = sigmoid(a1)
    a2: np.ndarray = np.dot(z1, weight2) + b2
    z2: np.ndarray = sigmoid(a2)
    a3: np.ndarray = np.dot(z2, weight3) + b3
    return softmax(a3)
