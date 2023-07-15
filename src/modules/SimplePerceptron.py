"""
main
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
        if np.issubsctype(weights.dtype.type, np.floating):
            raise UsageError(f"the type of weights must be a subtype of np.floating. ({function_position})")
        if np.isubdtype(weights.dtype.type, np.float64):
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
        if np.issubsctype(values.dtype.type, np.floating):
            raise UsageError(f"the type of input values must be a subtype of np.floating. ({function_position})")
        if np.isubdtype(values.dtype.type, np.float64):
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
            threshold: numbers.Real | np.floating
    ) -> None:
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_range(threshold, numbers.Real, "a threshold of ignition of a perceptron", function_position,
                       minimum=-1000.0, maximum=1000.0)
        object.__setattr__(self, "value", np.float64(threshold))


@dataclasses.dataclass(frozen=True)
class SimplePerceptron:
    """
    単純パーセプトロン
    """
    weights: Weights  # 重み配列
    threshold: Threshold  # 発火閾値

    def __init__(
            self,
            *,
            weights: Weights,
            threshold: Threshold
    ):
        function_position: Final[str] = (f"{inspect.currentframe().f_code.co_name}"
                                         f" of {self.__class__.__name__} in {__name__}")
        validate_type(weights, Weights, "weights of a perceptron", function_position)
        validate_type(threshold, Threshold, "a threshold of ignition of a perceptron", function_position)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "threshold", threshold)

    def output_value(self, input_values: InputValues) -> np.float64:
        """
        出力値
        Args:
            input_values(InputValues): 入力値
        Returns:
            np.float64: 出力値
        """
        return np.float64(0.0)
