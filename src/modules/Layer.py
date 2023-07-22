"""
propagation layers
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from src.modules.lower_layer_modules.Numbers_numpy import softmax


@dataclasses.dataclass
class MultiplicationLayer:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        forward propagation
        """
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        backward propagation
        """
        dx: np.ndarray = dout * self.y
        dy: np.ndarray = dout * self.x
        return dx, dy


class AdditionLayer:
    def __init__(self):
        pass

    @staticmethod
    def forward(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        forward propagation
        """
        return x + y

    @staticmethod
    def backward(dout: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        backward propagation
        """
        return dout * 1, dout * 1


class ReLULayer:

    def __init__(self):
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward propagation
        """
        self.mask = (x <= 0)
        out: np.ndarray = x.copy()
        out[self.mask] = np.float64(0.0)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        backward propagation
        """
        dout[self.mask] = np.float64(0.0)
        return dout


class SigmoidLayer:
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward propagation
        """
        out: np.ndarray = np.float64(1.0) / (np.float64(1.0) + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        backward propagation
        """
        dx: np.ndarray = dout * (1.0 - self.out) * self.out
        return dx


class AffineLayer:
    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.weight = weight
        self.bias = bias
        self.x: Optional[np.ndarray] = None
        self.dweight: Optional[np.ndarray] = None
        self.dbias: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        forward propagation
        """
        self.x = x
        out: np.ndarray = np.dot(x, self.weight) + self.bias
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        backward propagation
        """
        self.dweight = np.dot(self.x.T, dout)
        self.dbias = np.sum(dout, axis=0)
        return np.dot(dout, self.weight.T)


class SoftmaxWithLossLayer:
    def __init__(self):
        self.loss: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.float64:
        """
        forward propagation
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, _dout: np.ndarray = 1) -> np.ndarray:
        """
        backward propagation
        """
        batch_size: int = self.t.shape[0]
        return (self.y - self.t) / batch_size


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
            -np.sum(np.log(prediction[np.arange(batch_size), training_labels.astype(np.int64)] + np.float64(1.0e-7)))
            / np.float64(batch_size))
