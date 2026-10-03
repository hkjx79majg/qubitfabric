"""Core service surface for QubitFabric.

健康检查保持冻结基线行为；量子电路 IR 的构造、规范化、等价变换与
参数绑定委托给 :mod:`qubitfabric.circuit`，状态向量/密度矩阵仿真、
Pauli 期望值估计与参数移位梯度委托给 :mod:`qubitfabric.simulate`，
确定性变分优化委托给 :mod:`qubitfabric.optimize`，可暂停续算的优化
委托给 :mod:`qubitfabric.resumable`，不执行计算的资源预算准入估计
委托给 :mod:`qubitfabric.resources`，同一电路上多作业的批量期望值
委托给 :mod:`qubitfabric.batch`，带 LRU 缓存与稳定请求身份的期望值
估计委托给 :mod:`qubitfabric.cache`，可序列化的异步参数服务器
（参数状态创建与批量梯度更新）委托给 :mod:`qubitfabric.paramserver`，
错误类型在本模块公开。
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .batch import BatchExecutionError
from .batch import run_batch as _run_batch
from .cache import CacheStateError
from .cache import cached_expectation as _cached_expectation
from .circuit import (
    CircuitValidationError,
    ParameterBindingError,
    bind_parameters,
    normalize_circuit,
    simplify_circuit,
)
from .optimize import OptimizationError, optimize_circuit
from .paramserver import ParameterServerError
from .paramserver import apply_parameter_updates as _apply_parameter_updates
from .paramserver import create_parameter_state as _create_parameter_state
from .resources import ResourceEstimationError
from .resources import estimate_resources as _estimate_resources
from .resumable import RuntimeStateError
from .resumable import optimize_resumable as _optimize_resumable
from .simulate import SimulationError, estimate_expectation, estimate_gradient

__all__ = [
    "Service",
    "CircuitValidationError",
    "ParameterBindingError",
    "SimulationError",
    "OptimizationError",
    "ResourceEstimationError",
    "RuntimeStateError",
    "BatchExecutionError",
    "CacheStateError",
    "ParameterServerError",
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

    def cached_expectation(
        self,
        circuit: Any,
        observables: Any,
        values: Any = None,
        shots: Any = None,
        seed: Any = None,
        noise: Any = None,
        cache: Any = None,
        max_entries: Any = None,
    ) -> dict:
        """带 LRU 缓存与稳定请求身份的期望值估计，不改变 :meth:`expectation`。

        基础输入与 :meth:`expectation` 同含义、同校验顺序、同异常类型与
        量子位限制；``cache`` 为 ``{"version": 1, "entries"}`` 快照，
        ``max_entries`` 省略为 128。返回 ``{"result", "cache_hit",
        "request_id", "cache"}``：``result`` 与直接调用
        :meth:`expectation` 完全一致；``request_id`` 是请求身份的
        SHA-256（64 位小写十六进制）；``cache`` 仅含 JSON 原生类型，
        序列化往返后可跨进程传回复用。命中时核对摘要、返回独立副本并
        把条目移到首位；未命中时计算插入，超出容量淘汰末项。
        ``max_entries`` 或 ``cache`` 校验失败抛 :class:`CacheStateError`。
        不修改输入。
        """
        return _cached_expectation(
            circuit, observables, values=values, shots=shots, seed=seed, noise=noise,
            cache=cache, max_entries=max_entries,
        )

    def batch_expectation(self, circuit: Any, jobs: Any, max_concurrency: Any = None) -> dict:
        """在同一电路上批量执行多个独立期望值作业。

        公共电路先按现有语义规范化，失败抛 :class:`CircuitValidationError`；
        ``max_concurrency`` 省略为 1，显式值必须是排除 bool 的正整数；
        ``jobs`` 必须是非空数组，每项含唯一的非空字符串 ``id`` 与必填
        ``observables``，可携带与 :meth:`expectation` 同语义的
        ``values``/``shots``/``seed``/``noise``。返回 ``{"results",
        "summary"}``，results 按 jobs 输入顺序排列；单项绑定或仿真失败
        只影响该项，其余继续。请求级结构错误抛
        :class:`BatchExecutionError`。相同输入在任意合法并行度下结果
        内容一致，不修改输入。
        """
        return _run_batch(circuit, jobs, max_concurrency=max_concurrency)

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
        """可暂停、跨进程续算的确定性变分优化。

        基础输入与 :meth:`optimize` 相同；``step_budget`` 为正整数更新
        预算，每次调用至多执行这么多次参数更新。未结束时返回
        ``{"status": "paused", "result": None, "checkpoint", "progress"}``，
        检查点仅含 JSON 原生类型，序列化往返后可跨进程传回续算；
        结束时返回 ``{"status": "completed", "result",
        "checkpoint": None, "progress"}``，``result`` 与相同输入直接
        调用 :meth:`optimize` 完全一致。``step_budget`` 与
        ``checkpoint`` 校验失败抛 :class:`RuntimeStateError`；
        基础输入的校验顺序与异常和 :meth:`optimize` 相同。不修改输入。
        """
        return _optimize_resumable(
            circuit, terms, values, config,
            noise=noise, step_budget=step_budget, checkpoint=checkpoint,
        )

    def create_parameter_state(self, circuit: Any, values: Any) -> dict:
        """创建可序列化的异步参数服务器状态。

        电路规范化与参数绑定沿用 :meth:`create_circuit` 与 :meth:`bind`
        的语义，失败抛 :class:`CircuitValidationError` 或
        :class:`ParameterBindingError`。返回的状态仅含 JSON 原生类型：
        ``{"version", "parameters", "revision", "values", "updates"}``，
        修订号从零开始，``updates`` 为已接收更新的幂等记录（初始为空）。
        不修改输入。
        """
        return _create_parameter_state(circuit, values)

    def apply_parameter_updates(
        self,
        state: Any,
        learning_rate: Any,
        max_staleness: Any,
        updates: Any,
    ) -> dict:
        """按数组顺序合并一批基于不同参数版本计算的梯度更新。

        ``learning_rate`` 为正有限实数，``max_staleness`` 为非负整数，
        ``updates`` 为非空数组，每项含非空字符串 ``id``、非负整数
        ``base_revision`` 与恰好覆盖全部参数的有限梯度映射
        ``gradients``。``base_revision`` 不大于当前修订号且版本差不超过
        ``max_staleness`` 时接收，按 ``value - learning_rate * gradient``
        更新全部参数并记录内容摘要；过旧更新被拒绝（``stale``），引用
        未来修订的被拒绝（``future_revision``），均不改变状态；相同 id
        与内容再次出现返回 ``duplicate`` 及首次接收后的修订号，相同 id
        对应不同内容时整次调用失败。返回 ``{"state", "results"}``，
        ``state`` 为全新的最终状态（仅含 JSON 原生类型，序列化往返后
        继续使用结果一致），``results`` 与更新同序，每项含 ``id``、
        ``status``、``reason`` 与观察到的 ``revision``。状态、学习率、
        陈旧度、更新结构、摘要完整性、幂等冲突与非有限计算结果抛
        :class:`ParameterServerError`，异常时不返回部分状态。不修改输入。
        """
        return _apply_parameter_updates(state, learning_rate, max_staleness, updates)

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
