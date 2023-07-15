"""
論理回路
"""
from __future__ import annotations

import numbers

import numpy as np

from src.modules.SimplePerceptron import SimplePerceptron, Weights, Threshold, InputValues


def and_gate(x1: numbers.Real, x2: numbers.Real) -> np.float64:
    """
    ANDゲート
    Args:
        x1(numbers.Real): 入力1
        x2(numbers.Real): 入力2
    Returns:
        np.float64: 出力
    """
    perceptron_and: SimplePerceptron = SimplePerceptron(
        weights=Weights(weights=np.array([0.5, 0.5], dtype=np.float64)),
        threshold=Threshold(threshold=np.float64(0.7)))
    return perceptron_and.output_value(InputValues(values=np.array([x1, x2], dtype=np.float64)))
