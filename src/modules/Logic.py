"""
論理回路
"""
from __future__ import annotations

import numbers

import numpy as np

from src.modules.SimplePerceptron import SimplePerceptron, Weights, Threshold, InputValues


def and_gate(x1: numbers.Real, x2: numbers.Real) -> int:
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
        bias=Threshold(value=np.float64(-0.7)))
    return 1 if perceptron_and.ignite(InputValues(values=np.array([x1, x2], dtype=np.float64))) else 0


def nand_gate(x1: numbers.Real, x2: numbers.Real) -> int:
    """
    NANDゲート
    Args:
        x1(numbers.Real): 入力1
        x2(numbers.Real): 入力2
    Returns:
        np.float64: 出力
    """
    perceptron_nand: SimplePerceptron = SimplePerceptron(
        weights=Weights(weights=np.array([-0.5, -0.5], dtype=np.float64)),
        bias=Threshold(value=np.float64(0.7)))
    return 1 if perceptron_nand.ignite(InputValues(values=np.array([x1, x2], dtype=np.float64))) else 0


def or_gate(x1: numbers.Real, x2: numbers.Real) -> int:
    """
    ORゲート
    Args:
        x1(numbers.Real): 入力1
        x2(numbers.Real): 入力2
    Returns:
        np.float64: 出力
    """
    perceptron_or: SimplePerceptron = SimplePerceptron(
        weights=Weights(weights=np.array([0.5, 0.5], dtype=np.float64)),
        bias=Threshold(value=np.float64(-0.2)))
    return 1 if perceptron_or.ignite(InputValues(values=np.array([x1, x2], dtype=np.float64))) else 0


def xor_gate(x1: numbers.Real, x2: numbers.Real) -> int:
    """
    XORゲート
    Args:
        x1(numbers.Real): 入力1
        x2(numbers.Real): 入力2
    Returns:
        np.float64: 出力
    """
    return and_gate(nand_gate(x1, x2), or_gate(x1, x2))
