"""
section5 main うまくいかない
"""
from __future__ import annotations

from typing import MutableSequence, Mapping

import numpy as np

from src.sections.section5_TwoLayerNetwork import TwoLayerNetworkBackprop
from src.support.mnist import load_mnist


def backprop() -> None:
    """
    backprop
    """
    (training_images, training_labels), (test_images, test_labels) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list: MutableSequence[np.float64] = list()

    n_iteration: int = 10000
    train_size: int = training_images.shape[0]
    batch_size: int = 100
    learning_rate: np.float64 = np.float64(0.1)
    train_acc_list: MutableSequence[np.float64] = list()
    test_acc_list: MutableSequence[np.float64] = list()
    iteration_per_epoch: int = int(round(max(train_size / batch_size, 1)))

    network: TwoLayerNetworkBackprop = TwoLayerNetworkBackprop(
        input_layer_size=784, hidden_layer_size=50, output_layer_size=10)

    for iteration_index in range(n_iteration):
        batch_mask: np.ndarray = np.random.choice(train_size, batch_size)
        training_images_batch: np.ndarray = training_images[batch_mask]
        training_labels_batch: np.ndarray = training_labels[batch_mask]

        gradients: Mapping[str, np.ndarray] = network.gradient(training_images_batch, training_labels_batch)

        for key in ("W1", "b1", "W2", "b2"):
            network.parameters[key] -= learning_rate * gradients[key]

        loss: np.float64 = network.loss_function(training_images_batch, training_labels_batch)
        if np.isnan(loss):
            break
        train_loss_list.append(loss)
        print(f"iteration_index: {iteration_index}, loss: {loss}")

        if iteration_index % iteration_per_epoch == 0:
            train_acc: np.float64 = network.accuracy(training_images, training_labels)
            test_acc: np.float64 = network.accuracy(test_images, test_labels)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"train_acc: {train_acc}, test_acc: {test_acc}")


def gradient_check() -> None:
    """
    gradient check
    """
    (training_images, training_labels), (_test_images, _test_labels) = load_mnist(normalize=True, one_hot_label=True)
    network: TwoLayerNetworkBackprop = TwoLayerNetworkBackprop(
        input_layer_size=784, hidden_layer_size=50, output_layer_size=10)
    training_images_batch: np.ndarray = training_images[:3]
    training_labels_batch: np.ndarray = training_labels[:3]
    gradient_numerical: Mapping[str, np.ndarray] = network.numerical_gradient(
        training_images_batch, training_labels_batch)
    gradient_backprop: Mapping[str, np.ndarray] = network.gradient(training_images_batch, training_labels_batch)
    for key in gradient_numerical.keys():
        difference: np.float64 = np.average(np.abs(gradient_backprop[key] - gradient_numerical[key]))
        print(f"{key}: {difference}")
