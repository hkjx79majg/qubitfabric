"""Core service surface for QubitFabric.

健康检查保持冻结基线行为；量子电路 IR 的构造、规范化、等价变换与
参数绑定委托给 :mod:`qubitfabric.circuit`，状态向量/密度矩阵仿真、
Pauli 期望值估计与参数移位梯度委托给 :mod:`qubitfabric.simulate`，
确定性变分优化委托给 :mod:`qubitfabric.optimize`，不执行计算的资源
预算准入估计委托给 :mod:`qubitfabric.resources`，错误类型在本模块公开。
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
from .optimize import (
    OptimizationError,
    RuntimeStateError,
    optimize_circuit,
    optimize_resumable,
)
from .resources import ResourceEstimationError
from .resources import estimate_resources as _estimate_resources
from .simulate import SimulationError, estimate_expectation, estimate_gradient

__all__ = [
    "Service",
    "CircuitValidationError",
    "ParameterBindingError",
    "SimulationError",
    "OptimizationError",
    "RuntimeStateError",
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

    def optimize_resumable(
        self,
        circuit: Any,
        terms: Any,
        values: Any,
        config: Any,
        noise: Any = None,
        step_budget: Any = None,
        checkpoint: Any = None,
    ) -> dict:
        """可暂停、可跨进程续算的确定性变分优化。

        基础输入与 :meth:`optimize` 相同含义、相同校验顺序与异常；
        ``step_budget`` 为正整数，限制单次调用至多执行的更新次数。
        省略 ``checkpoint`` 从 iteration 0 开始，否则从检查点续算。
        未结束时返回 ``{"status": "paused", "result": None,
        "checkpoint", "progress"}``，``progress`` 含 ``iterations``、
        ``parameters``、``values``、``objective``、``gradient_norm``
        与从 iteration 0 起的完整 ``history``；结束时返回
        ``{"status": "completed", "result", "checkpoint": None}``，
        ``result`` 与相同输入直接调用 :meth:`optimize` 一致。
        ``step_budget`` 非法抛出 ``RuntimeStateError``
        （``invalid_step_budget``）；检查点非法为
        ``invalid_checkpoint``，来源不符为 ``checkpoint_mismatch``。
        不修改输入，不读写文件，检查点仅含 JSON 原生类型。
        """
        return optimize_resumable(
            circuit, terms, values, config,
            noise=noise, step_budget=step_budget, checkpoint=checkpoint,
        )

    def estimate_resources(self, circuit: Any, request: Any, budget: Any = None) -> dict:
        """不执行计算的资源预算准入估计。

        ``request.type`` 取 ``exact_expectation``/``sampled_expectation``/
        ``gradient``/``optimization``，字段沿用对应入口，无关字段非法。
        返回 ``{"qubit_count", "type", "representation", "state_elements",
        "state_bytes", "circuit_evaluations", "gate_applications",
        "total_shots", "runtime_supported", "admitted", "exceeded"}``。
        ``budget`` 省略为无限制；超限只写入 ``exceeded``，不抛异常。
        校验失败语义与对应计算入口一致，不修改输入。
        """
        return _estimate_resources(circuit, request, budget=budget)
