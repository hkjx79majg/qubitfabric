"""确定性变分优化：Pauli Hamiltonian 目标评估与经典参数更新。

目标函数为各项 Pauli observable 期望值与有限实系数乘积之和，按项的
输入顺序累加；梯度由 :func:`qubitfabric.simulate.estimate_gradient`
的精确参数移位结果按同一顺序加权得到。整个过程确定性执行：不支持
shots，不使用随机数，也不修改任何输入。

支持两种优化器（均按参数声明顺序计算加权梯度并同时更新全部参数）：

- ``gradient_descent``：``θ ← θ - learning_rate · g``；
- ``adam``：零初值一阶/二阶矩与标准偏差修正，
  ``m_t = β1 m_{t-1} + (1-β1) g_t``，
  ``v_t = β2 v_{t-1} + (1-β2) g_t²``，
  ``θ ← θ - lr · m̂_t / (√v̂_t + ε)``，
  beta1/beta2/epsilon 默认 0.9/0.999/1e-8。

每次评估记录一个历史状态（iteration 从 0 起连续编号，0 为初始点）：
``{"iteration", "values", "objective", "gradient_norm"}``；梯度欧氏
范数不大于 tolerance 时不做更新直接收敛，零参数电路零次更新即收敛。
"""

from __future__ import annotations

import math
from typing import Any

from .circuit import bind_parameters, normalize_circuit
from .simulate import (
    SimulationError,
    _MAX_QUBITS,
    _MAX_QUBITS_NOISY,
    _PAULI_CHARS,
    _validate_noise,
    estimate_gradient,
)

__all__ = ["OptimizationError", "run_optimization"]

_METHODS = ("gradient_descent", "adam")
_COMMON_OPTIONS = ("method", "learning_rate", "max_iterations", "tolerance")
_ADAM_ONLY_OPTIONS = ("beta1", "beta2", "epsilon")
_TERM_FIELDS = frozenset(("observable", "coefficient"))


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


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _validate_terms(terms: Any) -> list[float]:
    """校验 Hamiltonian 项的结构与系数，返回按输入顺序的系数列表。

    observable 的字符串/长度/字符合法性属于既有 observable 校验，
    在电路绑定得到量子位数后由 :func:`_validate_term_observables`
    按现有 SimulationError 语义报告。
    """
    if not isinstance(terms, list) or not terms:
        raise OptimizationError(
            "invalid_terms", "terms",
            "terms must be a non-empty array of Hamiltonian terms",
        )

    coefficients: list[float] = []
    for i, term in enumerate(terms):
        path = f"terms[{i}]"
        if not isinstance(term, dict):
            raise OptimizationError("invalid_term", path, f"term at {path} must be an object")

        for key in term:
            if not isinstance(key, str):
                raise OptimizationError("invalid_term", path, f"term at {path} has a non-string field name")

        if "observable" not in term:
            raise OptimizationError(
                "invalid_term", f"{path}.observable", f"missing field {path}.observable",
            )
        if "coefficient" not in term:
            raise OptimizationError(
                "invalid_term", f"{path}.coefficient", f"missing field {path}.coefficient",
            )

        coefficient = term["coefficient"]
        if not _is_number(coefficient) or not math.isfinite(float(coefficient)):
            raise OptimizationError(
                "invalid_term", f"{path}.coefficient",
                f"coefficient at {path}.coefficient must be a finite number",
            )

        unknown = sorted(key for key in term if key not in _TERM_FIELDS)
        if unknown:
            name = unknown[0]
            raise OptimizationError(
                "invalid_term", f"{path}.{name}", f"unknown term field {name!r} at {path}.{name}",
            )

        coefficients.append(_canon_float(float(coefficient)))

    return coefficients


def _validate_term_observables(terms: list[dict], qubit_count: int) -> list[str]:
    """按现有 observable 校验语义检查各项 Pauli 串，失败抛 SimulationError。"""
    observables: list[str] = []
    for i, term in enumerate(terms):
        path = f"terms[{i}].observable"
        observable = term["observable"]
        if not isinstance(observable, str):
            raise SimulationError(
                "invalid_observable", path, f"observable at {path} must be a string",
            )
        if len(observable) != qubit_count:
            raise SimulationError(
                "invalid_observable", path,
                f"observable at {path} must have length {qubit_count}",
            )
        if any(ch not in _PAULI_CHARS for ch in observable):
            raise SimulationError(
                "invalid_observable", path,
                f"observable at {path} may only contain I, X, Y, Z",
            )
        observables.append(observable)
    return observables


def _require_option(config: dict, name: str, path: str) -> Any:
    if name not in config:
        raise OptimizationError(
            "invalid_optimizer_option", path, f"missing required optimizer option {name!r}",
        )
    return config[name]


def _validate_config(config: Any) -> tuple[str, dict[str, Any]]:
    """校验优化配置，返回 ``(method, options)``；beta 参数缺省补默认值。"""
    if not isinstance(config, dict):
        raise OptimizationError(
            "invalid_optimizer_config", "config", "optimizer config must be an object",
        )
    for key in config:
        if not isinstance(key, str):
            raise OptimizationError(
                "invalid_optimizer_config", "config", "optimizer option names must be strings",
            )

    if "method" not in config:
        raise OptimizationError(
            "unsupported_optimizer", "config.method", "missing optimizer method",
        )
    method = config["method"]
    if not isinstance(method, str) or method not in _METHODS:
        raise OptimizationError(
            "unsupported_optimizer", "config.method", f"unsupported optimizer method {method!r}",
        )

    options: dict[str, Any] = {"method": method}

    raw_lr = _require_option(config, "learning_rate", "config.learning_rate")
    if not _is_number(raw_lr) or not math.isfinite(float(raw_lr)) or float(raw_lr) <= 0.0:
        raise OptimizationError(
            "invalid_optimizer_option", "config.learning_rate",
            "learning_rate must be a positive finite number",
        )
    options["learning_rate"] = float(raw_lr)

    raw_iterations = _require_option(config, "max_iterations", "config.max_iterations")
    if not _is_int(raw_iterations) or raw_iterations <= 0:
        raise OptimizationError(
            "invalid_optimizer_option", "config.max_iterations",
            "max_iterations must be a positive integer",
        )
    options["max_iterations"] = raw_iterations

    raw_tolerance = _require_option(config, "tolerance", "config.tolerance")
    if not _is_number(raw_tolerance) or not math.isfinite(float(raw_tolerance)) or float(raw_tolerance) < 0.0:
        raise OptimizationError(
            "invalid_optimizer_option", "config.tolerance",
            "tolerance must be a non-negative finite number",
        )
    options["tolerance"] = _canon_float(float(raw_tolerance))

    known = set(_COMMON_OPTIONS)
    if method == "adam":
        known.update(_ADAM_ONLY_OPTIONS)
        defaults = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
        for name in _ADAM_ONLY_OPTIONS:
            path = f"config.{name}"
            if name not in config:
                options[name] = defaults[name]
                continue
            value = config[name]
            if not _is_number(value) or not math.isfinite(float(value)):
                raise OptimizationError(
                    "invalid_optimizer_option", path, f"{name} must be a finite number",
                )
            number = float(value)
            if name in ("beta1", "beta2"):
                if number < 0.0 or number >= 1.0:
                    raise OptimizationError(
                        "invalid_optimizer_option", path, f"{name} must be in [0, 1)",
                    )
            elif number <= 0.0:
                raise OptimizationError(
                    "invalid_optimizer_option", path, f"{name} must be a positive finite number",
                )
            options[name] = number

    # 已知选项全部校验完后，未知字段按名字典序取首个。
    unknown = sorted(key for key in config if key not in known)
    if unknown:
        name = unknown[0]
        raise OptimizationError(
            "unknown_optimizer_option", f"config.{name}",
            f"unknown optimizer option {name!r} at config.{name}",
        )

    return method, options


# ---------------------------------------------------------------------------
# 目标评估与优化循环
# ---------------------------------------------------------------------------

def run_optimization(
    circuit: Any,
    terms: Any,
    values: Any = None,
    config: Any = None,
    noise: Any = None,
) -> dict:
    """对参数化电路做确定性变分优化，不修改输入。

    电路校验、参数绑定、observable、noise 与状态空间限制完全沿用现有
    语义，且任何此类错误都在第一次迭代更新之前抛出。
    """
    # 优化入口自身的校验先于量子侧校验。
    coefficients = _validate_terms(terms)
    method, options = _validate_config(config)

    normalized = normalize_circuit(circuit)
    bindings = {} if values is None else values
    bound = bind_parameters(normalized, bindings)
    qubit_count = bound["qubit_count"]
    parameters = list(normalized["parameters"])

    observables = _validate_term_observables(terms, qubit_count)
    p1, p2 = _validate_noise(noise)

    noisy = p1 != 0.0 or p2 != 0.0
    max_qubits = _MAX_QUBITS_NOISY if noisy else _MAX_QUBITS
    if qubit_count > max_qubits:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {max_qubits}-qubit simulation limit",
        )

    vector = [_canon_float(float(bindings[name])) for name in parameters]

    def evaluate(current: list[float]) -> tuple[float, list[float]]:
        current_bindings = {name: current[j] for j, name in enumerate(parameters)}
        gradient_result = estimate_gradient(
            normalized, observables, values=current_bindings, noise=noise,
        )
        results = gradient_result["results"]

        objective = 0.0
        for k, coefficient in enumerate(coefficients):
            objective = _canon_float(objective + coefficient * results[k]["expectation"])

        gradient: list[float] = []
        for name in parameters:
            weighted = 0.0
            for k, coefficient in enumerate(coefficients):
                weighted += coefficient * results[k]["gradients"][name]
            gradient.append(_canon_float(weighted))
        return objective, gradient

    learning_rate = options["learning_rate"]
    tolerance = options["tolerance"]
    max_iterations = options["max_iterations"]

    first_moments = [0.0] * len(parameters)
    second_moments = [0.0] * len(parameters)
    beta1 = options.get("beta1", 0.0)
    beta2 = options.get("beta2", 0.0)
    epsilon = options.get("epsilon", 0.0)

    history: list[dict] = []
    converged = False

    for iteration in range(max_iterations + 1):
        objective, gradient = evaluate(vector)
        gradient_norm = _canon_float(math.sqrt(sum(g * g for g in gradient)))
        current_values = {name: vector[j] for j, name in enumerate(parameters)}
        history.append({
            "iteration": iteration,
            "values": current_values,
            "objective": objective,
            "gradient_norm": gradient_norm,
        })

        if gradient_norm <= tolerance:
            converged = True
            break

        if iteration == max_iterations:
            break

        if method == "gradient_descent":
            vector = [
                _canon_float(x - learning_rate * g)
                for x, g in zip(vector, gradient)
            ]
        else:
            step = iteration + 1
            for j, g in enumerate(gradient):
                first_moments[j] = beta1 * first_moments[j] + (1.0 - beta1) * g
                second_moments[j] = beta2 * second_moments[j] + (1.0 - beta2) * g * g
            updated = []
            bias1 = 1.0 - beta1 ** step
            bias2 = 1.0 - beta2 ** step
            for j, x in enumerate(vector):
                m_hat = first_moments[j] / bias1
                v_hat = second_moments[j] / bias2
                updated.append(_canon_float(
                    x - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
                ))
            vector = updated

    final_state = history[-1]
    return {
        "converged": converged,
        "parameters": parameters,
        "iterations": len(history) - 1,
        "final_values": dict(final_state["values"]),
        "final_objective": final_state["objective"],
        "history": history,
    }
