"""Core service surface for QubitFabric.

健康检查保持冻结基线行为；量子电路 IR 的构造、规范化、等价变换与
参数绑定委托给 :mod:`qubitfabric.circuit`，状态向量/含噪密度矩阵
仿真、Pauli 期望值估计与参数移位梯度委托给 :mod:`qubitfabric.simulate`，
错误类型在本模块公开。
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
from .simulate import SimulationError, estimate_expectation, estimate_gradient

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
        noise: Any = None,
    ) -> dict:
        """状态向量/含噪密度矩阵仿真并估计各 Pauli observable 的期望值。

        省略 ``shots`` 返回精确期望值；给出 ``shots`` 则每项独立采样并
        返回正一/负一计数。可选的 ``noise`` 描述逐门局部退极化噪声，
        省略或概率全零时与无噪声结果一致。相同输入与 ``seed`` 结果
        完全一致，不修改输入。
        """
        return estimate_expectation(
            circuit, observables, values=values, shots=shots, seed=seed, noise=noise,
        )

    def gradient(self, circuit: Any, observables: Any, values: Any = None, noise: Any = None) -> dict:
        """精确参数移位梯度：各 observable 对全部声明参数的导数。

        含义与校验语义和精确 :meth:`expectation` 一致；可选的 ``noise``
        描述逐门局部退极化噪声，含噪时基准值与导数都来自含噪仿真。
        返回 ``{"qubit_count", "parameters", "results"}``，每项结果为
        ``{"observable", "expectation", "gradients"}``。不修改输入。
        """
        return estimate_gradient(circuit, observables, values=values, noise=noise)
