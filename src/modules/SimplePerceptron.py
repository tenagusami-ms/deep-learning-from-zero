"""
simple linear perceptron
"""
from __future__ import annotations

import dataclasses
import inspect
import numbers
from typing import Final

import numpy as np

from src.modules.lower_layer_modules.Exceptions import UsageError
from src.modules.lower_layer_modules.Validation import validate_type, validate_range


@dataclasses.dataclass(frozen=True)
class Weights:
    """
    パーセプトロンの重み
    """
    weights: np.ndarray

    def __init__(
            self,
            *,
            weights: np.ndarray
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(weights, np.ndarray, "weights of a perceptron", function_position)
        if weights.ndim != 1:
            raise UsageError(f"the dimension of weights must be 1. ({function_position})")
        if not np.issubdtype(weights.dtype, np.number):
            raise UsageError(f"the type of each data in weights must be a subtype of np.number."
                             f" ({function_position})")
        if np.issubdtype(weights.dtype, np.float64):
            object.__setattr__(self, "weights", weights.astype(np.float64))
        else:
            object.__setattr__(self, "weights", weights)


class InputValues:
    """
    パーセプトロンの入力
    """
    values: np.ndarray

    def __init__(
            self,
            *,
            values: np.ndarray
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(values, np.ndarray, "input values of a perceptron", function_position)
        if values.ndim != 1:
            raise UsageError(f"the dimension of input values must be 1. ({function_position})")
        if not np.issubdtype(values.dtype, np.number):
            raise UsageError(f"the type of input values must be a subtype of np.number. ({function_position})")
        if np.issubdtype(values.dtype, np.float64):
            object.__setattr__(self, "values", values.astype(np.float64))
        else:
            object.__setattr__(self, "values", values)


@dataclasses.dataclass(frozen=True)
class Threshold:
    """
    パーセプトロンの発火閾値
    """
    value: np.float64  # 発火閾値

    def __init__(
            self,
            *,
            value: numbers.Real | np.floating
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_range(value, numbers.Real, "a threshold of ignition of a perceptron", function_position,
                       minimum=-1000.0, maximum=1000.0)
        object.__setattr__(self, "value", np.float64(value))


@dataclasses.dataclass(frozen=True)
class SimplePerceptron:
    """
    単純パーセプトロン
    """
    weights: Weights  # 重み配列
    bias: Threshold  # バイアス

    def __init__(
            self,
            *,
            weights: Weights,
            bias: Threshold
    ):
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(weights, Weights, "weights of a perceptron", function_position)
        validate_type(bias, Threshold, "the bias of a perceptron", function_position)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)

    def ignite(self, input_values: InputValues) -> bool:
        """
        発火
        Args:
            input_values(InputValues): 入力値
        Returns:
            bool: 発火したならTrue
        """
        return self.output_value(input_values) >= np.float64(0.0)

    def output_value(self, input_values: InputValues) -> np.float64:
        """
        出力値
        Args:
            input_values(InputValues): 入力値
        Returns:
            np.float64: 出力値
        """
        return np.dot(self.weights.weights, input_values.values) + self.bias.value
