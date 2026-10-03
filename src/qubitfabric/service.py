"""Core service surface for QubitFabric.

健康检查保持冻结基线行为；量子电路 IR 的构造、规范化、等价变换与
参数绑定委托给 :mod:`qubitfabric.circuit`，状态向量/密度矩阵仿真、
Pauli 期望值估计与参数移位梯度委托给 :mod:`qubitfabric.simulate`，
确定性变分优化委托给 :mod:`qubitfabric.optimize`，错误类型在本模块公开。
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
from .optimize import OptimizationError, optimize_circuit
from .resources import ResourceEstimationError, estimate_resources
from .simulate import SimulationError, estimate_expectation, estimate_gradient

__all__ = [
    "Service",
    "CircuitValidationError",
    "ParameterBindingError",
    "SimulationError",
    "OptimizationError",
    "ResourceEstimationError",
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
        """无噪声状态向量仿真并估计各 Pauli observable 的期望值。

        省略 ``shots`` 返回精确期望值；给出 ``shots`` 则每项独立采样并
        返回正一/负一计数。相同输入与 ``seed`` 结果完全一致，不修改输入。
        可选 ``noise`` 描述逐门局部退极化噪声，省略时行为不变。
        """
        return estimate_expectation(
            circuit, observables, values=values, shots=shots, seed=seed, noise=noise,
        )

    def gradient(self, circuit: Any, observables: Any, values: Any = None, noise: Any = None) -> dict:
        """精确参数移位梯度：各 observable 对全部声明参数的导数。

        含义与校验语义和精确 :meth:`expectation` 一致；返回
        ``{"qubit_count", "parameters", "results"}``，每项结果为
        ``{"observable", "expectation", "gradients"}``。不修改输入。
        可选 ``noise`` 描述逐门局部退极化噪声，省略时行为不变。
        """
        return estimate_gradient(circuit, observables, values=values, noise=noise)

    def optimize(
        self,
        circuit: Any,
        terms: Any,
        values: Any,
        config: Any,
        noise: Any = None,
    ) -> dict:
        """确定性变分优化：精确目标评估与 gradient_descent / adam 经典更新。

        ``terms`` 为 Hamiltonian 项数组，每项含匹配量子位数的 Pauli
        ``observable`` 与有限实数 ``coefficient``；目标值按输入顺序累加
        期望值与系数的乘积。``values`` 为完整且无未知名称的初始参数；
        ``config`` 给出 ``method``、``learning_rate``、``max_iterations``、
        ``tolerance`` 及 adam 可选的 ``beta1``/``beta2``/``epsilon``。
        返回 ``{"converged", "iterations", "parameters", "final_values",
        "final_objective", "history"}``。不修改输入，不使用随机数；
        可选 ``noise`` 沿用逐门局部退极化语义。
        """
        return optimize_circuit(circuit, terms, values, config, noise=noise)

    def estimate_resources(self, circuit: Any, request: Any, budget: Any = None) -> dict:
        """不执行计算的资源预算准入。

        ``request`` 的 ``type`` 取 ``exact_expectation``、
        ``sampled_expectation``、``gradient``、``optimization``，字段
        沿用对应入口；返回状态表示与规模、电路评估次数、门应用次数、
        采样总数、运行时支持标志、超限预算键与准入结论。``budget``
        省略为无限制。不修改输入，超限不抛异常；非法请求/预算抛
        :class:`ResourceEstimationError`，其他错误沿用对应入口语义。
        """
        return estimate_resources(circuit, request, budget)
