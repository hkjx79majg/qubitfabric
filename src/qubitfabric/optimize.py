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

:func:`optimize_resumable` 在同一优化循环上提供可暂停、可跨进程续算的
入口：调用方以 ``step_budget`` 限制单次调用执行的更新次数，未收敛也未
达到 ``max_iterations`` 时返回 ``paused`` 状态与纯 JSON 检查点；检查点
序列化往返后可在任意进程中继续，任意分段的最终数值、参数顺序与历史都
与单次 :func:`optimize_circuit` 完全一致。预算与检查点校验在基础输入
之后依次进行，失败抛出 :class:`RuntimeStateError`。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .circuit import bind_parameters, normalize_circuit
from .simulate import (
    _NOISE_FIELDS,
    _validate_noise,
    _validate_observables,
    estimate_gradient,
)

__all__ = ["OptimizationError", "RuntimeStateError", "optimize_circuit", "optimize_resumable"]

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


class RuntimeStateError(ValueError):
    """可续算优化的运行状态校验失败。

    ``code`` 为 ``invalid_step_budget``（预算不是正整数）、
    ``invalid_checkpoint``（检查点结构、类型、数值或内部状态非法）或
    ``checkpoint_mismatch``（检查点结构合法但并非由相同的 circuit、
    terms、初始 values、config 和 noise 产生）；``path`` 分别为
    ``"step_budget"`` 与 ``"checkpoint"``。
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


class _RunState:
    """优化循环的运行状态（内存形态）；检查点是其 JSON 序列化形态。"""

    __slots__ = (
        "current", "objective", "gradients", "norm",
        "iterations", "history", "first_moment", "second_moment",
    )

    def __init__(self, current, objective, gradients, norm, iterations,
                 history, first_moment, second_moment):
        self.current = current
        self.objective = objective
        self.gradients = gradients
        self.norm = norm
        self.iterations = iterations
        self.history = history
        self.first_moment = first_moment
        self.second_moment = second_moment


def _prepare_run(circuit, terms, values, config, noise):
    """按 optimize 的顺序校验基础输入，返回运行上下文。"""
    normalized = normalize_circuit(circuit)
    initial = {} if values is None else values
    bind_parameters(normalized, initial)
    observables, coefficients = _validate_terms(terms)
    _validate_observables(observables, normalized["qubit_count"])
    noise_probs = _validate_noise(noise)
    cfg = _validate_config(config)
    parameters = list(normalized["parameters"])
    start = {name: float(initial[name]) for name in parameters}
    return normalized, parameters, start, observables, coefficients, noise_probs, cfg


def _make_evaluator(normalized, observables, coefficients, parameters, noise):
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
    return _evaluate


def _initial_state(parameters, start, evaluate) -> _RunState:
    """iteration 0：与 optimize 相同的初始点评估。"""
    current = dict(start)
    objective, gradients = evaluate(current)
    norm = _norm(gradients, parameters)
    return _RunState(
        current=current,
        objective=objective,
        gradients=gradients,
        norm=norm,
        iterations=0,
        history=[_snapshot(0, current, objective, norm)],
        first_moment={name: 0.0 for name in parameters},
        second_moment={name: 0.0 for name in parameters},
    )


def _step(state: _RunState, cfg: dict, parameters: list[str], evaluate) -> None:
    """执行一次参数更新，评估新点并追加历史。"""
    learning_rate = cfg["learning_rate"]
    if cfg["method"] == "gradient_descent":
        current = {
            name: state.current[name] - learning_rate * state.gradients[name]
            for name in parameters
        }
    else:  # adam
        beta1 = cfg["beta1"]
        beta2 = cfg["beta2"]
        epsilon = cfg["epsilon"]
        step = state.iterations + 1
        correction1 = 1.0 - beta1 ** step
        correction2 = 1.0 - beta2 ** step
        updated: dict[str, float] = {}
        for name in parameters:
            grad = state.gradients[name]
            state.first_moment[name] = beta1 * state.first_moment[name] + (1.0 - beta1) * grad
            state.second_moment[name] = beta2 * state.second_moment[name] + (1.0 - beta2) * grad * grad
            m_hat = state.first_moment[name] / correction1
            v_hat = state.second_moment[name] / correction2
            updated[name] = state.current[name] - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
        current = updated

    state.current = current
    state.iterations += 1
    state.objective, state.gradients = evaluate(current)
    state.norm = _norm(state.gradients, parameters)
    state.history.append(_snapshot(state.iterations, current, state.objective, state.norm))


def _result(state: _RunState, parameters: list[str], cfg: dict) -> dict:
    return {
        "converged": state.norm <= cfg["tolerance"],
        "iterations": state.iterations,
        "parameters": list(parameters),
        "final_values": {name: _canon_float(state.current[name]) for name in parameters},
        "final_objective": _canon_float(state.objective),
        "history": state.history,
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
    normalized, parameters, start, observables, coefficients, _, cfg = _prepare_run(
        circuit, terms, values, config, noise,
    )
    evaluate = _make_evaluator(normalized, observables, coefficients, parameters, noise)
    state = _initial_state(parameters, start, evaluate)
    while state.norm > cfg["tolerance"] and state.iterations < cfg["max_iterations"]:
        _step(state, cfg, parameters, evaluate)
    return _result(state, parameters, cfg)


# ---------------------------------------------------------------------------
# 可暂停、可续算的优化
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION = 1
_CHECKPOINT_FIELDS = frozenset((
    "version", "fingerprint", "iterations", "values", "gradients",
    "objective", "gradient_norm", "history", "optimizer",
))
_HISTORY_FIELDS = frozenset(("iteration", "values", "objective", "gradient_norm"))
_OPTIMIZER_FIELDS = frozenset(("first_moment", "second_moment"))


def _fingerprint(normalized, observables, coefficients, start, noise_probs, cfg) -> str:
    """基础输入的稳定指纹：同一 circuit、terms、初始 values、config 和
    noise 产生的检查点才允许续算。"""
    payload = {
        "circuit": normalized,
        "terms": [
            {"observable": observable, "coefficient": coefficient}
            for observable, coefficient in zip(observables, coefficients)
        ],
        "values": {name: _canon_float(start[name]) for name in start},
        "noise": dict(zip(_NOISE_FIELDS, noise_probs)),
        "config": cfg,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _build_checkpoint(state: _RunState, parameters: list[str], fingerprint: str) -> dict:
    """把运行状态导出为纯 JSON 原生类型的检查点（不共享可变对象）。"""
    return {
        "version": _CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "iterations": state.iterations,
        "values": {name: state.current[name] for name in parameters},
        "gradients": {name: state.gradients[name] for name in parameters},
        "objective": state.objective,
        "gradient_norm": state.norm,
        "history": [
            {
                "iteration": entry["iteration"],
                "values": dict(entry["values"]),
                "objective": entry["objective"],
                "gradient_norm": entry["gradient_norm"],
            }
            for entry in state.history
        ],
        "optimizer": {
            "first_moment": dict(state.first_moment),
            "second_moment": dict(state.second_moment),
        },
    }


def _invalid_checkpoint(message: str) -> None:
    raise RuntimeStateError("invalid_checkpoint", "checkpoint", message)


def _checkpoint_number(value: Any, what: str) -> float:
    number = _finite(value)
    if number is None:
        _invalid_checkpoint(f"checkpoint {what} must be a finite number")
    return number


def _checkpoint_mapping(value: Any, what: str) -> dict[str, float]:
    if not isinstance(value, dict):
        _invalid_checkpoint(f"checkpoint {what} must be an object")
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _invalid_checkpoint(f"checkpoint {what} field names must be strings")
        result[key] = _checkpoint_number(item, f"{what}.{key}")
    return result


def _checkpoint_snapshot(entry: Any, index: int) -> dict:
    what = f"history[{index}]"
    if not isinstance(entry, dict):
        _invalid_checkpoint(f"checkpoint {what} must be an object")
    for key in entry:
        if not isinstance(key, str):
            _invalid_checkpoint(f"checkpoint {what} field names must be strings")
    unknown = sorted(key for key in entry if key not in _HISTORY_FIELDS)
    if unknown:
        _invalid_checkpoint(f"unknown checkpoint field {what}.{unknown[0]}")
    for field in ("iteration", "values", "objective", "gradient_norm"):
        if field not in entry:
            _invalid_checkpoint(f"missing checkpoint field {what}.{field}")
    iteration = entry["iteration"]
    if not _is_int(iteration) or iteration < 0:
        _invalid_checkpoint(f"checkpoint {what}.iteration must be a non-negative integer")
    return {
        "iteration": iteration,
        "values": _checkpoint_mapping(entry["values"], f"{what}.values"),
        "objective": _checkpoint_number(entry["objective"], f"{what}.objective"),
        "gradient_norm": _checkpoint_number(entry["gradient_norm"], f"{what}.gradient_norm"),
    }


def _restore_checkpoint(checkpoint: Any, parameters: list[str], cfg: dict, fingerprint: str) -> _RunState:
    """校验检查点并还原运行状态。

    结构、类型、数值与内部一致性非法时抛出 ``invalid_checkpoint``；
    结构合法但指纹不符时抛出 ``checkpoint_mismatch``。
    """
    if not isinstance(checkpoint, dict):
        _invalid_checkpoint("checkpoint must be an object")
    for key in checkpoint:
        if not isinstance(key, str):
            _invalid_checkpoint("checkpoint field names must be strings")
    unknown = sorted(key for key in checkpoint if key not in _CHECKPOINT_FIELDS)
    if unknown:
        _invalid_checkpoint(f"unknown checkpoint field {unknown[0]!r}")
    for field in ("version", "fingerprint", "iterations", "values", "gradients",
                  "objective", "gradient_norm", "history", "optimizer"):
        if field not in checkpoint:
            _invalid_checkpoint(f"missing checkpoint field {field!r}")

    version = checkpoint["version"]
    if not _is_int(version) or version != _CHECKPOINT_VERSION:
        _invalid_checkpoint(f"unsupported checkpoint version {version!r}")
    if not isinstance(checkpoint["fingerprint"], str):
        _invalid_checkpoint("checkpoint fingerprint must be a string")
    iterations = checkpoint["iterations"]
    if not _is_int(iterations) or iterations < 0:
        _invalid_checkpoint("checkpoint iterations must be a non-negative integer")
    values = _checkpoint_mapping(checkpoint["values"], "values")
    gradients = _checkpoint_mapping(checkpoint["gradients"], "gradients")
    objective = _checkpoint_number(checkpoint["objective"], "objective")
    norm = _checkpoint_number(checkpoint["gradient_norm"], "gradient_norm")

    raw_history = checkpoint["history"]
    if not isinstance(raw_history, list):
        _invalid_checkpoint("checkpoint history must be an array")
    snapshots = [_checkpoint_snapshot(entry, i) for i, entry in enumerate(raw_history)]

    optimizer = checkpoint["optimizer"]
    if not isinstance(optimizer, dict):
        _invalid_checkpoint("checkpoint optimizer must be an object")
    for key in optimizer:
        if not isinstance(key, str):
            _invalid_checkpoint("checkpoint optimizer field names must be strings")
    unknown = sorted(key for key in optimizer if key not in _OPTIMIZER_FIELDS)
    if unknown:
        _invalid_checkpoint(f"unknown checkpoint field optimizer.{unknown[0]}")
    for field in ("first_moment", "second_moment"):
        if field not in optimizer:
            _invalid_checkpoint(f"missing checkpoint field optimizer.{field}")
    first_moment = _checkpoint_mapping(optimizer["first_moment"], "optimizer.first_moment")
    second_moment = _checkpoint_mapping(optimizer["second_moment"], "optimizer.second_moment")

    # 状态内部一致性：iterations、history 与当前状态必须互相吻合。
    if len(snapshots) != iterations + 1:
        _invalid_checkpoint("checkpoint history length contradicts iterations")
    for i, snap in enumerate(snapshots):
        if snap["iteration"] != i:
            _invalid_checkpoint("checkpoint history iterations must count up from 0")
    last = snapshots[-1]
    if (last["values"] != values or last["objective"] != objective
            or last["gradient_norm"] != norm):
        _invalid_checkpoint("checkpoint state contradicts its last history entry")

    if checkpoint["fingerprint"] != fingerprint:
        raise RuntimeStateError(
            "checkpoint_mismatch", "checkpoint",
            "checkpoint was not produced from the same circuit, terms, "
            "values, config and noise",
        )

    # 指纹一致后仍按当前输入防御性核对参数集合与终止条件。
    names = set(parameters)
    if (set(values) != names or set(gradients) != names
            or set(first_moment) != names or set(second_moment) != names
            or any(set(snap["values"]) != names for snap in snapshots)):
        _invalid_checkpoint("checkpoint parameter set contradicts the circuit")
    if iterations >= cfg["max_iterations"]:
        _invalid_checkpoint("checkpoint iterations already reached max_iterations")
    if norm <= cfg["tolerance"]:
        _invalid_checkpoint("checkpoint gradient_norm already satisfies tolerance")

    return _RunState(
        current={name: values[name] for name in parameters},
        objective=objective,
        gradients={name: gradients[name] for name in parameters},
        norm=norm,
        iterations=iterations,
        history=[
            _snapshot(
                snap["iteration"],
                {name: snap["values"][name] for name in parameters},
                snap["objective"],
                snap["gradient_norm"],
            )
            for snap in snapshots
        ],
        first_moment={name: first_moment[name] for name in parameters},
        second_moment={name: second_moment[name] for name in parameters},
    )


def optimize_resumable(
    circuit: Any,
    terms: Any,
    values: Any,
    config: Any,
    noise: Any = None,
    step_budget: Any = None,
    checkpoint: Any = None,
) -> dict:
    """可暂停、可跨进程续算的确定性变分优化，不修改输入，不读写文件。

    基础输入（circuit、terms、values、config、noise）的含义、校验顺序与
    异常和 :func:`optimize_circuit` 完全一致；其后依次校验
    ``step_budget``（正整数，否则 ``invalid_step_budget``）与
    ``checkpoint``（非法抛出 ``invalid_checkpoint``，指纹不符抛出
    ``checkpoint_mismatch``）。

    省略 ``checkpoint`` 时从 iteration 0 开始（与 optimize 相同的初始
    评估），否则从检查点还原状态继续；单次调用至多执行 ``step_budget``
    次更新。梯度范数达到 ``tolerance`` 或累计更新达到
    ``config.max_iterations`` 时返回
    ``{"status": "completed", "result", "checkpoint": None}``，其中
    ``result`` 与相同输入直接调用 optimize 的返回一致；否则返回
    ``{"status": "paused", "result": None, "checkpoint", "progress"}``，
    ``progress`` 含 ``iterations``、``parameters``、``values``、
    ``objective``、``gradient_norm`` 与从 iteration 0 起的完整
    ``history``。检查点仅由 JSON 原生类型组成，序列化往返后可续算。
    """
    normalized, parameters, start, observables, coefficients, noise_probs, cfg = _prepare_run(
        circuit, terms, values, config, noise,
    )
    if not _is_int(step_budget) or step_budget < 1:
        raise RuntimeStateError(
            "invalid_step_budget", "step_budget",
            "step_budget must be a positive integer",
        )
    fingerprint = _fingerprint(normalized, observables, coefficients, start, noise_probs, cfg)
    evaluate = _make_evaluator(normalized, observables, coefficients, parameters, noise)
    if checkpoint is None:
        state = _initial_state(parameters, start, evaluate)
    else:
        state = _restore_checkpoint(checkpoint, parameters, cfg, fingerprint)

    updates = 0
    while (state.norm > cfg["tolerance"]
           and state.iterations < cfg["max_iterations"]
           and updates < step_budget):
        _step(state, cfg, parameters, evaluate)
        updates += 1

    if state.norm <= cfg["tolerance"] or state.iterations >= cfg["max_iterations"]:
        return {
            "status": "completed",
            "result": _result(state, parameters, cfg),
            "checkpoint": None,
        }
    return {
        "status": "paused",
        "result": None,
        "checkpoint": _build_checkpoint(state, parameters, fingerprint),
        "progress": {
            "iterations": state.iterations,
            "parameters": list(parameters),
            "values": {name: _canon_float(state.current[name]) for name in parameters},
            "objective": _canon_float(state.objective),
            "gradient_norm": _canon_float(state.norm),
            "history": state.history,
        },
    }
