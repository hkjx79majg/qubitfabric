"""Core service surface for QubitFabric.

健康检查保持冻结基线行为；量子电路 IR 的构造、规范化、等价变换与
参数绑定委托给 :mod:`qubitfabric.circuit`，无噪声状态向量仿真与
Pauli 期望值估计委托给 :mod:`qubitfabric.simulation`，错误类型在本模块公开。
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .circuit import (
    CircuitValidationError,
    ParameterBindingError,
    bind_parameters,
    normalize_circuit,
    simplify_circuit,
)
from .simulation import SimulationError, estimate_expectation

__all__ = [
    "Service",
    "CircuitValidationError",
    "ParameterBindingError",
    "SimulationError",
]


class Service:
    """QubitFabric 服务：健康检查与量子电路 IR 能力。"""

    name = "qubitfabric"
    version = __version__

    def health(self) -> dict[str, str]:
        return {"status": "ok", "service": self.name, "version": self.version}

    def create_circuit(self, data: Any) -> dict:
        """提交 JSON 兼容对象构造电路，返回规范化的新对象，不修改输入。"""
        return normalize_circuit(data)

    def simplify(self, circuit: Any) -> dict:
        """对规范化电路做确定性等价变换。

        返回 ``{"circuit", "removed_operations", "merged_operations"}``。
        """
        return simplify_circuit(circuit)

    def bind(self, circuit: Any, values: Any) -> dict:
        """把参数名到有限实数的映射绑定进电路，返回参数为空的新电路。"""
        return bind_parameters(circuit, values)

    def expectation(
        self,
        circuit: Any,
        observables: Any,
        values: Any = None,
        shots: Any = None,
        seed: Any = None,
    ) -> dict:
        """无噪声状态向量仿真，按输入顺序返回各 Pauli 观测量的期望值。

        省略 ``shots`` 时返回精确期望值；给出 ``shots`` 时对每项独立采样，
        返回 ±1 计数与由计数得到的期望值。相同输入与 ``seed`` 结果完全一致。
        """
        return estimate_expectation(circuit, observables, values, shots, seed)
