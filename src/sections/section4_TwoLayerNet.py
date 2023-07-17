"""
two-layer neural network (section 4)
"""
from __future__ import annotations

import dataclasses
from typing import MutableMapping, Mapping, Callable

import numpy as np

from src.modules.lower_layer_modules.Numbers_numpy import sigmoid, softmax


@dataclasses.dataclass(frozen=True)
class TwoLayerNetwork:
    """
    two-layer neural network
    """
    parameters: MutableMapping[str, np.ndarray]

    def __init__(
            self,
            input_layer_size: int,
            hidden_layer_size: int,
            output_layer_size: int,
            weight_init_std: np.float64 = 0.01
    ):
        parameters: MutableMapping[str, np.ndarray] = dict()
        object.__setattr__(self, "parameters", parameters)
        self.parameters["W1"] = weight_init_std * np.random.randn(input_layer_size, hidden_layer_size)
        self.parameters["b1"] = np.zeros(hidden_layer_size)
        self.parameters["W2"] = weight_init_std * np.random.randn(hidden_layer_size, output_layer_size)
        self.parameters["b2"] = np.zeros(output_layer_size)

    def predict(self, input_layer: np.ndarray) -> np.ndarray:
        """
        predict
        """
        weight1, weight2 = self.parameters["W1"], self.parameters["W2"]
        bias1, bias2 = self.parameters["b1"], self.parameters["b2"]
        hidden_layer: np.ndarray = sigmoid(np.dot(input_layer, weight1) + bias1)
        return softmax(np.dot(hidden_layer, weight2) + bias2)

    def loss_function(self, input_layer: np.ndarray, training_labels: np.ndarray) -> np.float64:
        """
        loss_function
        """
        prediction: np.ndarray = self.predict(input_layer)
        return cross_entropy_error(prediction, training_labels)

    def accuracy(self, input_layer: np.ndarray, training_labels: np.ndarray) -> np.float64:
        """
        accuracy
        """
        prediction: np.ndarray = self.predict(input_layer)
        prediction_index: np.ndarray = np.argmax(prediction, axis=1)
        training_labels: np.ndarray = np.argmax(training_labels, axis=1)
        return np.float64(np.sum(prediction_index == training_labels) / input_layer.shape[0])

    def numerical_gradient(
            self,
            input_layer: np.ndarray,
            training_labels: np.ndarray
    ) -> Mapping[str, np.ndarray]:
        """
        numerical_gradient
        """
        loss_function: Callable[[np.ndarray], np.float64] = (
            lambda weight: self.loss_function(input_layer, training_labels))
        gradients: MutableMapping[str, np.ndarray] = dict()
        gradients["W1"] = numerical_gradient(loss_function, self.parameters["W1"])
        gradients["b1"] = numerical_gradient(loss_function, self.parameters["b1"])
        gradients["W2"] = numerical_gradient(loss_function, self.parameters["W2"])
        gradients["b2"] = numerical_gradient(loss_function, self.parameters["b2"])
        return gradients


def cross_entropy_error(prediction: np.ndarray, training_labels: np.ndarray) -> np.float64:
    """
    cross-entropy error
    """
    if prediction.ndim == 1:
        training_labels = training_labels.reshape(1, training_labels.size)
        prediction = prediction.reshape(1, prediction.size)
    if training_labels.size == prediction.size:
        training_labels = training_labels.argmax(axis=1)
    batch_size: int = prediction.shape[0]
    return (
            -np.sum(np.log(prediction[np.arange(batch_size), training_labels.astype(np.int8)] + np.float64(1.0e-7)))
            / batch_size)


def numerical_gradient(function: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    """
    numerical gradient
    """
    h: np.float64 = np.float64(1.0e-4)  # 0.0001
    gradient: np.ndarray = np.zeros_like(x)
    it: np.nditer = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        index: np.ndarray = it.multi_index
        tmp_val: np.float64 = x[index]
        x[index] = tmp_val + h
        fxh1: np.ndarray = function(x)
        x[index] = tmp_val - h
        fxh2: np.ndarray = function(x)
        gradient[index] = (fxh1 - fxh2) / (np.float64(2.0) * h)
        x[index] = tmp_val
        it.iternext()
    return gradient
