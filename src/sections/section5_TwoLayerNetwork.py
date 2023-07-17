"""
section4 main
"""
from __future__ import annotations

import dataclasses
from typing import MutableMapping, Callable, Mapping

import numpy as np

from src.modules.Layer import AffineLayer, ReLULayer, SoftmaxWithLossLayer


@dataclasses.dataclass(frozen=True)
class TwoLayerNetworkBackprop:
    """
    two-layer neural network with backward propagation
    """
    parameters: MutableMapping[str, np.ndarray]
    layers: MutableMapping
    last_layer: SoftmaxWithLossLayer

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

        object.__setattr__(self, "layers", dict())
        self.layers["Affine1"] = AffineLayer(self.parameters["W1"], self.parameters["b1"])
        self.layers["Relu1"] = ReLULayer()
        self.layers["Affine2"] = AffineLayer(self.parameters["W2"], self.parameters["b2"])
        object.__setattr__(self, "last_layer", SoftmaxWithLossLayer())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        loss
        """
        y: np.ndarray = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        accuracy
        """
        y: np.ndarray = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t: np.ndarray = np.argmax(t, axis=1)
        accuracy: np.ndarray = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> Mapping[str, np.ndarray]:
        """
        numerical gradient
        """
        loss_weight: Callable[[np.ndarray], np.ndarray] = lambda weight: self.loss(x, t)
        gradients: MutableMapping[str, np.ndarray] = dict()
        gradients["W1"] = numerical_gradient(loss_weight, self.parameters["W1"])
        gradients["b1"] = numerical_gradient(loss_weight, self.parameters["b1"])
        gradients["W2"] = numerical_gradient(loss_weight, self.parameters["W2"])
        gradients["b2"] = numerical_gradient(loss_weight, self.parameters["b2"])
        return gradients


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
