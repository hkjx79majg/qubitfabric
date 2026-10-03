"""资源预算准入估计：不执行任何计算，仅按入口语义估算资源需求。

:func:`estimate_resources` 复用各计算入口的校验语义（电路、参数绑定、
observables、噪声、shots/seed、Hamiltonian 项与优化器配置），在不仿真
的前提下给出确定性的资源画像：状态表示、状态元素与字节数、电路评估
次数、门作用次数与采样总次数。

``request.type`` 与字段对应现有入口：

- ``exact_expectation`` → 精确期望值（``observables``/``values``/``noise``）；
- ``sampled_expectation`` → 采样期望值（另含 ``shots``/``seed``）；
- ``gradient`` → 参数移位梯度（``observables``/``values``/``noise``）；
- ``optimization`` → 变分优化（``terms``/``values``/``config``/``noise``）。

无关字段非法；``observables``、采样模式的 ``shots`` 以及优化模式的
``terms``/``values``/``config`` 为必填。

无正概率噪声时状态表示为 ``state_vector``（2^qubit_count 元素），否则为
``density_matrix``（4^qubit_count 元素），每元素 16 字节；两种表示的
量子位上限依次为 20、10，超出时 ``runtime_supported`` 为 false（只写入
结果，不抛异常）。电路评估次数：两种期望各 1 次，梯度 1+2r，优化最多
(max_iterations+1)×(1+2r)，其中 r 为规范化后非零系数的参数化 rx/rz
数量；门作用次数为评估次数乘操作数；采样模式总 shots 为
shots×observable 数，其余类型为 0。

``budget`` 省略表示无限制；否则仅可含 ``max_state_bytes``、
``max_circuit_evaluations``、``max_gate_applications``、
``max_total_shots``，值须为排除 bool 的非负整数。``exceeded`` 按此顺序
列出超限键；``admitted`` 为 ``runtime_supported`` 且 ``exceeded`` 为空。
非法请求与预算抛 :class:`ResourceEstimationError`（code 分别为
``invalid_resource_request``、``invalid_budget``，path 指向首个问题）；
其余校验错误沿用对应入口的异常类型、code、path 与顺序。不修改输入。
"""

from __future__ import annotations

from typing import Any

from .circuit import bind_parameters, normalize_circuit
from .optimize import _validate_config, _validate_terms
from .simulate import (
    _validate_noise,
    _validate_observables,
    _validate_seed,
    _validate_shots,
)

__all__ = ["ResourceEstimationError", "estimate_resources"]

_BYTES_PER_ELEMENT = 16
_MAX_QUBITS_STATE_VECTOR = 20
_MAX_QUBITS_DENSITY_MATRIX = 10

_BUDGET_KEYS = (
    "max_state_bytes",
    "max_circuit_evaluations",
    "max_gate_applications",
    "max_total_shots",
)
_BUDGET_KEY_SET = frozenset(_BUDGET_KEYS)

_REQUEST_FIELDS = {
    "exact_expectation": frozenset(("type", "observables", "values", "noise")),
    "sampled_expectation": frozenset(("type", "observables", "values", "shots", "seed", "noise")),
    "gradient": frozenset(("type", "observables", "values", "noise")),
    "optimization": frozenset(("type", "terms", "values", "config", "noise")),
}
_REQUIRED_FIELDS = {
    "exact_expectation": ("observables",),
    "sampled_expectation": ("observables", "shots"),
    "gradient": ("observables",),
    "optimization": ("terms", "values", "config"),
}


class ResourceEstimationError(ValueError):
    """资源估计入口的请求/预算校验失败。

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


# ---------------------------------------------------------------------------
# request / budget 结构校验
# ---------------------------------------------------------------------------

def _validate_request(request: Any) -> str:
    """校验请求结构，返回请求类型。

    依次检查：对象形态与字段名 → ``type`` 合法 → 无未知字段（多个按
    字典序取首个）→ 必填字段齐全。
    """
    if not isinstance(request, dict):
        raise ResourceEstimationError(
            "invalid_resource_request", "request", "request must be an object"
        )
    for key in request:
        if not isinstance(key, str):
            raise ResourceEstimationError(
                "invalid_resource_request", "request", "request field names must be strings"
            )
    request_type = request.get("type")
    if request_type not in _REQUEST_FIELDS:
        raise ResourceEstimationError(
            "invalid_resource_request", "request.type",
            f"unknown request type {request_type!r} at request.type",
        )
    unknown = sorted(key for key in request if key not in _REQUEST_FIELDS[request_type])
    if unknown:
        name = unknown[0]
        raise ResourceEstimationError(
            "invalid_resource_request", f"request.{name}",
            f"unknown request field {name!r} at request.{name}",
        )
    for field in _REQUIRED_FIELDS[request_type]:
        if field not in request:
            raise ResourceEstimationError(
                "invalid_resource_request", f"request.{field}",
                f"missing field request.{field}",
            )
    return request_type


def _validate_budget(budget: Any) -> dict[str, int]:
    """校验预算，返回规范化后的上限映射；None 表示无限制。"""
    if budget is None:
        return {}
    if not isinstance(budget, dict):
        raise ResourceEstimationError(
            "invalid_budget", "budget", "budget must be an object"
        )
    for key in budget:
        if not isinstance(key, str):
            raise ResourceEstimationError(
                "invalid_budget", "budget", "budget field names must be strings"
            )
    limits: dict[str, int] = {}
    for key in _BUDGET_KEYS:
        if key not in budget:
            continue
        value = budget[key]
        if not _is_int(value) or value < 0:
            raise ResourceEstimationError(
                "invalid_budget", f"budget.{key}",
                f"budget at budget.{key} must be a non-negative integer",
            )
        limits[key] = value
    unknown = sorted(key for key in budget if key not in _BUDGET_KEY_SET)
    if unknown:
        name = unknown[0]
        raise ResourceEstimationError(
            "invalid_budget", f"budget.{name}",
            f"unknown budget field {name!r} at budget.{name}",
        )
    return limits


# ---------------------------------------------------------------------------
# 各请求类型的入口语义校验（顺序与对应计算入口一致，不做状态空间检查）
# ---------------------------------------------------------------------------

def _validate_expectation_request(
    circuit: Any, request: dict, sampled: bool
) -> tuple[dict, float, float, list[str], int | None]:
    values = request.get("values")
    normalized = normalize_circuit(circuit)
    bind_parameters(normalized, {} if values is None else values)
    observables = _validate_observables(request.get("observables"), normalized["qubit_count"])
    p1, p2 = _validate_noise(request.get("noise"))
    shots = None
    if sampled:
        shots = _validate_shots(request.get("shots"))
        _validate_seed(request.get("seed"), shots)
    return normalized, p1, p2, observables, shots


def _validate_gradient_request(circuit: Any, request: dict) -> tuple[dict, float, float]:
    values = request.get("values")
    normalized = normalize_circuit(circuit)
    bind_parameters(normalized, {} if values is None else values)
    _validate_observables(request.get("observables"), normalized["qubit_count"])
    p1, p2 = _validate_noise(request.get("noise"))
    return normalized, p1, p2


def _validate_optimization_request(circuit: Any, request: dict) -> tuple[dict, float, float, dict]:
    values = request.get("values")
    normalized = normalize_circuit(circuit)
    bind_parameters(normalized, {} if values is None else values)
    observables, _coefficients = _validate_terms(request.get("terms"))
    _validate_observables(observables, normalized["qubit_count"])
    p1, p2 = _validate_noise(request.get("noise"))
    config = _validate_config(request.get("config"))
    return normalized, p1, p2, config


# ---------------------------------------------------------------------------
# 资源估计
# ---------------------------------------------------------------------------

def estimate_resources(circuit: Any, request: Any, budget: Any = None) -> dict:
    """估算请求的资源需求并按预算准入，不执行任何计算，不修改输入。

    返回 ``{"qubit_count", "type", "representation", "state_elements",
    "state_bytes", "circuit_evaluations", "gate_applications",
    "total_shots", "runtime_supported", "admitted", "exceeded"}``，
    全部为 JSON 原生类型且结果确定。
    """
    request_type = _validate_request(request)
    limits = _validate_budget(budget)

    shots: int | None = None
    observable_count = 0
    max_iterations = 0
    if request_type in ("exact_expectation", "sampled_expectation"):
        normalized, p1, p2, observables, shots = _validate_expectation_request(
            circuit, request, request_type == "sampled_expectation"
        )
        observable_count = len(observables)
    elif request_type == "gradient":
        normalized, p1, p2 = _validate_gradient_request(circuit, request)
    else:  # optimization
        normalized, p1, p2, config = _validate_optimization_request(circuit, request)
        max_iterations = config["max_iterations"]

    qubit_count = normalized["qubit_count"]
    noisy = p1 != 0.0 or p2 != 0.0
    if noisy:
        representation = "density_matrix"
        state_elements = 1 << (2 * qubit_count)
        runtime_supported = qubit_count <= _MAX_QUBITS_DENSITY_MATRIX
    else:
        representation = "state_vector"
        state_elements = 1 << qubit_count
        runtime_supported = qubit_count <= _MAX_QUBITS_STATE_VECTOR
    state_bytes = state_elements * _BYTES_PER_ELEMENT

    # 规范化后零系数参数表达式已折叠为常量，剩余字典角度即非零系数参数化旋转。
    parameterized = sum(
        1 for op in normalized["operations"] if isinstance(op.get("angle"), dict)
    )
    if request_type in ("exact_expectation", "sampled_expectation"):
        evaluations = 1
    elif request_type == "gradient":
        evaluations = 1 + 2 * parameterized
    else:  # optimization
        evaluations = (max_iterations + 1) * (1 + 2 * parameterized)

    gate_applications = evaluations * len(normalized["operations"])
    total_shots = shots * observable_count if shots is not None else 0

    estimates = {
        "max_state_bytes": state_bytes,
        "max_circuit_evaluations": evaluations,
        "max_gate_applications": gate_applications,
        "max_total_shots": total_shots,
    }
    exceeded = [key for key in _BUDGET_KEYS if key in limits and estimates[key] > limits[key]]

    return {
        "qubit_count": qubit_count,
        "type": request_type,
        "representation": representation,
        "state_elements": state_elements,
        "state_bytes": state_bytes,
        "circuit_evaluations": evaluations,
        "gate_applications": gate_applications,
        "total_shots": total_shots,
        "runtime_supported": runtime_supported,
        "admitted": runtime_supported and not exceeded,
        "exceeded": exceeded,
    }
