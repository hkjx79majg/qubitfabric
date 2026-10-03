"""确定性变分优化：精确目标评估与经典更新规则的组合。

:func:`optimize_circuit` 接收电路 IR、Hamiltonian 项、初始参数与优化器
配置，目标值为各项 ``coefficient * ⟨observable⟩`` 按输入顺序累加之和，
梯度由 :func:`qubitfabric.simulate.estimate_gradient` 的精确参数移位
规则给出，按参数声明顺序加权并同时更新。全程不使用随机数，不修改输入。

优化器：

- ``gradient_descent``：``θ ← θ - learning_rate * ∇``；
- ``adam``：零初值一阶/二阶矩与标准偏差修正，``beta1``/``beta2``/
  ``epsilon`` 默认 ``0.9``/``0.999``/``1e-8``。

``learning_rate`` 为正有限数，``max_iterations`` 为正整数，
``tolerance`` 为非负有限数，三者均为必填；梯度欧氏范数不大于
``tolerance`` 时停止且 ``converged`` 为 true，达到更新上限则为 false。

校验顺序：电路（:class:`CircuitValidationError`）→ 初始参数绑定
（:class:`ParameterBindingError`）→ Hamiltonian 项结构
（:class:`OptimizationError`）→ observable 内容（:class:`SimulationError`）
→ noise（:class:`SimulationError`）→ 优化器配置
（:class:`OptimizationError`）→ 状态空间上限（:class:`SimulationError`，
随首次目标评估发生，先于任何参数更新）。

:func:`_prepare`、:func:`_initial_state`、:func:`_advance` 与
:func:`_finalize` 把校验、初始评估、按预算推进与结果汇总拆开，
供 :mod:`qubitfabric.resumable` 的可暂停续算入口复用同一实现。
"""

from __future__ import annotations

import math
from typing import Any

from .circuit import bind_parameters, normalize_circuit
from .simulate import _validate_noise, _validate_observables, estimate_gradient

__all__ = ["OptimizationError", "optimize_circuit"]

_METHODS = ("gradient_descent", "adam")
_TERM_FIELDS = frozenset(("observable", "coefficient"))
_CONFIG_FIELDS = frozenset(
    ("method", "learning_rate", "max_iterations", "tolerance", "beta1", "beta2", "epsilon")
)
_DEFAULT_BETA1 = 0.9
_DEFAULT_BETA2 = 0.999
_DEFAULT_EPSILON = 1e-8


class OptimizationError(ValueError):
    """优化入口校验失败。

    ``code`` 为稳定的机器可读错误码，``path`` 指向输入中出错的位置，
    语义与 :class:`qubitfabric.circuit.CircuitValidationError` 一致。
    """

    def __init__(self, code: str, path: str, message: str | None = None) -> None:
        self.code = code
        self.path = path
        if message is None:
            message = f"{code} at {path}"
        super().__init__(message)


def _is_int(value: Any) -> bool:
    """JSON 整数；bool 是 int 的子类，但不算数值。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """JSON 实数（int 或 float），排除布尔值。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canon_float(value: float) -> float:
    """统一浮点形态：抹掉 -0.0。"""
    if value == 0.0:
        return 0.0
    return value


def _finite(value: Any) -> float | None:
    """是有限实数时返回 float，否则返回 None。"""
    if not _is_number(value):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


# ---------------------------------------------------------------------------
# Hamiltonian 项
# ---------------------------------------------------------------------------

def _validate_terms(terms: Any) -> tuple[list[str], list[float]]:
    """校验 Hamiltonian 项结构，返回 ``(observables, coefficients)``。

    observable 的内容（长度、字符集）沿用现有
    :class:`qubitfabric.simulate.SimulationError` 校验，不在此展开。
    """
    if not isinstance(terms, list) or not terms:
        raise OptimizationError(
            "invalid_terms", "terms",
            "terms must be a non-empty array of Hamiltonian terms",
        )
    observables: list[str] = []
    coefficients: list[float] = []
    for i, term in enumerate(terms):
        path = f"terms[{i}]"
        if not isinstance(term, dict):
            raise OptimizationError("invalid_term", path, f"term at {path} must be an object")
        for key in term:
            if not isinstance(key, str):
                raise OptimizationError(
                    "invalid_term", path, f"term field names at {path} must be strings"
                )
        if "observable" not in term:
            raise OptimizationError(
                "invalid_term", f"{path}.observable", f"missing field {path}.observable"
            )
        observable = term["observable"]
        if not isinstance(observable, str):
            raise OptimizationError(
                "invalid_term", f"{path}.observable",
                f"observable at {path}.observable must be a Pauli string",
            )
        if "coefficient" not in term:
            raise OptimizationError(
                "invalid_term", f"{path}.coefficient", f"missing field {path}.coefficient"
            )
        coefficient = _finite(term["coefficient"])
        if coefficient is None:
            raise OptimizationError(
                "invalid_term", f"{path}.coefficient",
                f"coefficient at {path}.coefficient must be a finite number",
            )
        unknown = sorted(key for key in term if key not in _TERM_FIELDS)
        if unknown:
            name = unknown[0]
            raise OptimizationError(
                "invalid_term", f"{path}.{name}", f"unknown term field {name!r} at {path}.{name}"
            )
        observables.append(observable)
        coefficients.append(_canon_float(coefficient))
    return observables, coefficients


# ---------------------------------------------------------------------------
# 优化器配置
# ---------------------------------------------------------------------------

def _validate_config(config: Any) -> dict:
    """校验优化器配置，返回带默认值的完整配置。"""
    if not isinstance(config, dict):
        raise OptimizationError(
            "invalid_optimizer_config", "config", "optimizer config must be an object"
        )
    for key in config:
        if not isinstance(key, str):
            raise OptimizationError(
                "invalid_optimizer_config", "config", "optimizer config field names must be strings"
            )

    method = config.get("method")
    if method not in _METHODS:
        raise OptimizationError(
            "unsupported_optimizer", "config.method",
            f"unsupported optimizer method {method!r} at config.method",
        )

    unknown = sorted(key for key in config if key not in _CONFIG_FIELDS)
    if unknown:
        name = unknown[0]
        raise OptimizationError(
            "unknown_optimizer_option", f"config.{name}",
            f"unknown optimizer option {name!r} at config.{name}",
        )

    learning_rate = _finite(config.get("learning_rate"))
    if learning_rate is None or learning_rate <= 0.0:
        raise OptimizationError(
            "invalid_optimizer_option", "config.learning_rate",
            "learning_rate at config.learning_rate must be a positive finite number",
        )

    max_iterations = config.get("max_iterations")
    if not _is_int(max_iterations) or max_iterations < 1:
        raise OptimizationError(
            "invalid_optimizer_option", "config.max_iterations",
            "max_iterations at config.max_iterations must be a positive integer",
        )

    tolerance = _finite(config.get("tolerance"))
    if tolerance is None or tolerance < 0.0:
        raise OptimizationError(
            "invalid_optimizer_option", "config.tolerance",
            "tolerance at config.tolerance must be a non-negative finite number",
        )

    beta1 = _DEFAULT_BETA1
    if "beta1" in config:
        value = _finite(config["beta1"])
        if value is None or value < 0.0 or value >= 1.0:
            raise OptimizationError(
                "invalid_optimizer_option", "config.beta1",
                "beta1 at config.beta1 must be a finite number in [0, 1)",
            )
        beta1 = value

    beta2 = _DEFAULT_BETA2
    if "beta2" in config:
        value = _finite(config["beta2"])
        if value is None or value < 0.0 or value >= 1.0:
            raise OptimizationError(
                "invalid_optimizer_option", "config.beta2",
                "beta2 at config.beta2 must be a finite number in [0, 1)",
            )
        beta2 = value

    epsilon = _DEFAULT_EPSILON
    if "epsilon" in config:
        value = _finite(config["epsilon"])
        if value is None or value <= 0.0:
            raise OptimizationError(
                "invalid_optimizer_option", "config.epsilon",
                "epsilon at config.epsilon must be a positive finite number",
            )
        epsilon = value

    return {
        "method": method,
        "learning_rate": learning_rate,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
    }


# ---------------------------------------------------------------------------
# 优化循环
# ---------------------------------------------------------------------------

def _snapshot(iteration: int, values: dict[str, float], objective: float, norm: float) -> dict:
    return {
        "iteration": iteration,
        "values": {name: _canon_float(value) for name, value in values.items()},
        "objective": _canon_float(objective),
        "gradient_norm": _canon_float(norm),
    }


def _norm(gradients: dict[str, float], parameters: list[str]) -> float:
    return math.sqrt(sum(gradients[name] * gradients[name] for name in parameters))


def _prepare(
    circuit: Any,
    terms: Any,
    values: Any,
    config: Any,
    noise: Any,
) -> tuple[dict, dict, list[str], list[float], tuple[float, float], dict, list[str], Any]:
    """按固定顺序校验基础优化输入，返回优化循环所需的全部派生量。

    校验顺序：电路 → 初始参数绑定 → Hamiltonian 项结构 → observable
    内容 → noise → 优化器配置；状态空间上限随首次目标评估发生。
    """
    normalized = normalize_circuit(circuit)
    initial = {} if values is None else values
    bind_parameters(normalized, initial)
    observables, coefficients = _validate_terms(terms)
    _validate_observables(observables, normalized["qubit_count"])
    noise_probs = _validate_noise(noise)
    cfg = _validate_config(config)

    parameters = list(normalized["parameters"])

    def _evaluate(point: dict[str, float]) -> tuple[float, dict[str, float]]:
        result = estimate_gradient(normalized, observables, values=point, noise=noise)
        objective = 0.0
        gradients = {name: 0.0 for name in parameters}
        for k, coefficient in enumerate(coefficients):
            item = result["results"][k]
            objective += coefficient * item["expectation"]
            for name in parameters:
                gradients[name] += coefficient * item["gradients"][name]
        return objective, gradients

    return normalized, initial, observables, coefficients, noise_probs, cfg, parameters, _evaluate


def _initial_state(parameters: list[str], initial: dict, evaluate: Any) -> dict:
    """iteration 0 的循环状态：初始点评估，未做任何更新。"""
    current = {name: float(initial[name]) for name in parameters}
    objective, gradients = evaluate(current)
    norm = _norm(gradients, parameters)
    return {
        "iterations": 0,
        "current": current,
        "gradients": gradients,
        "objective": objective,
        "norm": norm,
        "history": [_snapshot(0, current, objective, norm)],
        "first_moment": {name: 0.0 for name in parameters},
        "second_moment": {name: 0.0 for name in parameters},
    }


def _advance(state: dict, cfg: dict, parameters: list[str], evaluate: Any, budget: int) -> None:
    """就地推进 ``state``，至多执行 ``budget`` 次参数更新。

    梯度范数不大于 ``tolerance`` 或累计更新达到 ``max_iterations`` 时
    提前停止；数值与未分段执行完全相同。
    """
    method = cfg["method"]
    learning_rate = cfg["learning_rate"]
    max_iterations = cfg["max_iterations"]
    tolerance = cfg["tolerance"]
    beta1 = cfg["beta1"]
    beta2 = cfg["beta2"]
    epsilon = cfg["epsilon"]

    iterations = state["iterations"]
    current = state["current"]
    gradients = state["gradients"]
    objective = state["objective"]
    norm = state["norm"]
    history = state["history"]
    first_moment = state["first_moment"]
    second_moment = state["second_moment"]
    converged = norm <= tolerance
    remaining = budget

    while not converged and iterations < max_iterations and remaining > 0:
        if method == "gradient_descent":
            current = {
                name: current[name] - learning_rate * gradients[name]
                for name in parameters
            }
        else:  # adam
            step = iterations + 1
            correction1 = 1.0 - beta1 ** step
            correction2 = 1.0 - beta2 ** step
            updated: dict[str, float] = {}
            for name in parameters:
                grad = gradients[name]
                first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * grad
                second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * grad * grad
                m_hat = first_moment[name] / correction1
                v_hat = second_moment[name] / correction2
                updated[name] = current[name] - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
            current = updated

        iterations += 1
        objective, gradients = evaluate(current)
        norm = _norm(gradients, parameters)
        history.append(_snapshot(iterations, current, objective, norm))
        converged = norm <= tolerance
        remaining -= 1

    state["iterations"] = iterations
    state["current"] = current
    state["gradients"] = gradients
    state["objective"] = objective
    state["norm"] = norm


def _finalize(state: dict, cfg: dict, parameters: list[str]) -> dict:
    """把循环状态汇总为 ``optimize`` 的返回对象。"""
    return {
        "converged": state["norm"] <= cfg["tolerance"],
        "iterations": state["iterations"],
        "parameters": parameters,
        "final_values": {name: _canon_float(state["current"][name]) for name in parameters},
        "final_objective": _canon_float(state["objective"]),
        "history": state["history"],
    }


def optimize_circuit(
    circuit: Any,
    terms: Any,
    values: Any,
    config: Any,
    noise: Any = None,
) -> dict:
    """确定性变分优化，不修改输入，不使用随机数。

    返回 ``{"converged", "iterations", "parameters", "final_values",
    "final_objective", "history"}``：``parameters`` 保持声明顺序；
    ``history`` 保存初始点（iteration 0）和每次更新后的状态，每项为
    ``{"iteration", "values", "objective", "gradient_norm"}``。
    """
    _, initial, _, _, _, cfg, parameters, evaluate = _prepare(circuit, terms, values, config, noise)
    state = _initial_state(parameters, initial, evaluate)
    _advance(state, cfg, parameters, evaluate, cfg["max_iterations"])
    return _finalize(state, cfg, parameters)
