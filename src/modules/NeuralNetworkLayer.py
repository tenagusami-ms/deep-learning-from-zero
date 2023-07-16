"""
a layer in a neural network
"""
from __future__ import annotations

import dataclasses
import inspect
from typing import Final

import numpy as np

from src.modules.lower_layer_modules.Exceptions import UsageError
from src.modules.lower_layer_modules.Validation import validate_type


@dataclasses.dataclass(frozen=True)
class Weights:
    """
    weights of a layer in a neural network
    """
    weights: np.ndarray

    def __init__(
            self,
            *,
            weights: np.ndarray
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(weights, np.ndarray, "weights of a layer in a neural network", function_position)
        if weights.ndim != 2:
            raise UsageError(f"the dimension of weights must be 2. ({function_position})")
        if not np.issubdtype(weights.dtype, np.number):
            raise UsageError(f"the type of each data in weights must be a subtype of np.number."
                             f" ({function_position})")
        if np.issubdtype(weights.dtype, np.float64):
            object.__setattr__(self, "weights", weights.astype(np.float64))
        else:
            object.__setattr__(self, "weights", weights)


@dataclasses.dataclass(frozen=True)
class Biases:
    """
    biases of a layer in a neural network
    """
    biases: np.ndarray

    def __init__(
            self,
            *,
            biases: np.ndarray
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(biases, np.ndarray, "biases of a layer in a neural network", function_position)
        if biases.ndim != 1:
            raise UsageError(f"the dimension of biases must be 1. ({function_position})")
        if not np.issubdtype(biases.dtype, np.number):
            raise UsageError(f"the type of each data in biases must be a subtype of np.number."
                             f" ({function_position})")
        if np.issubdtype(biases.dtype, np.float64):
            object.__setattr__(self, "biases", biases.astype(np.float64))
        else:
            object.__setattr__(self, "biases", biases)


@dataclasses.dataclass(frozen=True)
class InputValues:
    """
    input values of a layer in a neural network
    """
    values: np.ndarray

    def __init__(
            self,
            *,
            values: np.ndarray
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(values, np.ndarray, "input values of a layer in a neural network", function_position)
        if values.ndim != 1:
            raise UsageError(f"the dimension of input values must be 1. ({function_position})")
        if not np.issubdtype(values.dtype, np.number):
            raise UsageError(f"the type of input values must be a subtype of np.number. ({function_position})")
        if np.issubdtype(values.dtype, np.float64):
            object.__setattr__(self, "values", values.astype(np.float64))
        else:
            object.__setattr__(self, "values", values)


@dataclasses.dataclass(frozen=True)
class Layer:
    """
    a layer in a neural network
    """
    weights: Weights
    biases: Biases

    def __init__(
            self,
            *,
            weights: Weights,
            biases: Biases
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(weights, Weights, "weights of a layer in a neural network", function_position)
        validate_type(biases, Biases, "biases of a layer in a neural network", function_position)
        if weights.weights.shape[1] != biases.biases.shape[0]:
            raise UsageError(f"the number of columns of weights must be equal to the number of rows of biases."
                             f" ({function_position})")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "biases", biases)

    def output_values(self, input_values: InputValues) -> np.ndarray:
        """
        output values of a layer in a neural network
        """
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(input_values, InputValues, "input values of a layer in a neural network", function_position)
        if self.weights.weights.shape[0] != input_values.values.shape[0]:
            raise UsageError(f"the number of rows of weights must be equal to the number of elements of input values."
                             f" ({function_position})")
        return np.dot(self.weights.weights, input_values.values) + self.biases.biases
