"""
logical gates
"""
from __future__ import annotations

from src.modules.Logic import and_gate, nand_gate, or_gate, xor_gate


def gates() -> None:
    """
    gates
    """
    _and1: int = and_gate(0, 0)
    _and2: int = and_gate(0, 1)
    _and3: int = and_gate(1, 0)
    _and4: int = and_gate(1, 1)
    _nand1: int = nand_gate(0, 0)
    _nand2: int = nand_gate(0, 1)
    _nand3: int = nand_gate(1, 0)
    _nand4: int = nand_gate(1, 1)
    _or1: int = or_gate(0, 0)
    _or2: int = or_gate(0, 1)
    _or3: int = or_gate(1, 0)
    _or4: int = or_gate(1, 1)
    _xor1: int = xor_gate(0, 0)
    _xor2: int = xor_gate(0, 1)
    _xor3: int = xor_gate(1, 0)
    _xor4: int = xor_gate(1, 1)
    pass
