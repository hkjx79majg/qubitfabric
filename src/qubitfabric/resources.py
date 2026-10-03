"""不执行计算的资源预算准入。

:func:`estimate_resources` 与既有四个计算入口共用校验链，但只做静态
核算、绝不构造状态或采样：

- ``exact_expectation`` / ``sampled_expectation``：对应
  :meth:`qubitfabric.service.Service.expectation`；
- ``gradient``：对应 :meth:`qubitfabric.service.Service.gradient`；
- ``optimization``：对应 :meth:`qubitfabric.service.Service.optimize`。

状态表示：噪声概率全为零时用 ``state_vector``（``2**qubit_count``
个元素），存在正概率噪声时用 ``density_matrix``（``4**qubit_count``
个元素），每个元素按 16 字节计。两种表示的量子位上限分别为 20、10，
超出仅令 ``runtime_supported`` 为 false（不抛异常）。

电路评估次数：两种期望各计 1 次；梯度计 ``1 + 2r`` 次（基线一次，
每个出现在参数化 rx/rz 线性角中的旋转门两次移位）；优化最多计
``(max_iterations + 1) * (1 + 2r)`` 次（初始点加每次更新各一次梯度
评估的上限，不考虑提前收敛）。``r`` 为规范化后系数非零的参数化
rx、rz 旋转门数量。``gate_applications`` 为评估次数乘以电路操作数；
采样模式 ``total_shots`` 为 ``shots`` 乘以 observable 数量。

``budget`` 省略表示无限制；否则只能含 ``max_state_bytes``、
``max_circuit_evaluations``、``max_gate_applications``、
``max_total_shots``，值为非布尔、非负整数。``exceeded`` 按此固定
顺序列出被超过的预算键。准入结果 ``admitted`` 为
``runtime_supported`` 且没有任何超限项。
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

_MAX_QUBITS_STATE_VECTOR = 20
_MAX_QUBITS_DENSITY_MATRIX = 10
_BYTES_PER_ELEMENT = 16

_REQUEST_TYPES = (
    "exact_expectation",
    "sampled_expectation",
    "gradient",
    "optimization",
)
_TYPE_FIELDS = {
    "exact_expectation": frozenset(("type", "observables", "values", "noise")),
    "sampled_expectation": frozenset(
        ("type", "observables", "values", "shots", "seed", "noise")
    ),
    "gradient": frozenset(("type", "observables", "values", "noise")),
    "optimization": frozenset(("type", "terms", "values", "config", "noise")),
}
_REQUIRED_FIELDS = {
    "exact_expectation": ("observables",),
    "sampled_expectation": ("observables", "shots"),
    "gradient": ("observables",),
    "optimization": ("terms", "config"),
}
_BUDGET_FIELDS = (
    "max_state_bytes",
    "max_circuit_evaluations",
    "max_gate_applications",
    "max_total_shots",
)
_BUDGET_FIELD_SET = frozenset(_BUDGET_FIELDS)


class ResourceEstimationError(ValueError):
    """资源准入入口自身的校验失败。

    非法请求 code 为 ``invalid_resource_request``，非法预算为
    ``invalid_budget``；``path`` 指向首个问题位置。电路、绑定、
    observable、噪声、Hamiltonian 项与优化器配置的内容错误沿用对应
    计算入口的异常类型、code 与 path。
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


def _count_parameterized_rotations(normalized: dict) -> int:
    """规范化后系数非零的参数化 rx、rz 旋转门数量 r。

    规范化已把零系数参数角折叠为常量 float，故 angle 仍为 dict 即代表
    系数非零的参数化旋转。
    """
    count = 0
    for op in normalized["operations"]:
        if op["gate"] in ("rx", "rz") and isinstance(op["angle"], dict):
            count += 1
    return count


def _validate_budget(budget: Any) -> dict[str, int] | None:
    """校验预算对象，返回各项上限的拷贝；省略（None）表示无限制。"""
    if budget is None:
        return None
    if not isinstance(budget, dict):
        raise ResourceEstimationError("invalid_budget", "$", "budget must be an object")
    for key in budget:
        if not isinstance(key, str) or key not in _BUDGET_FIELD_SET:
            path = key if isinstance(key, str) else "$"
            raise ResourceEstimationError("invalid_budget", path, f"unknown budget field {path!r}")
    limits: dict[str, int] = {}
    for field in _BUDGET_FIELDS:
        if field not in budget:
            continue
        value = budget[field]
        if not _is_int(value) or value < 0:
            raise ResourceEstimationError(
                "invalid_budget",
                field,
                f"budget field {field} must be a non-negative integer",
            )
        limits[field] = value
    return limits


def _validate_request(request: Any) -> dict:
    """校验资源请求对象的结构（type、字段集合与必填字段），返回请求本身。

    字段内容沿用对应计算入口的校验，不在此展开。结构问题的报告顺序：
    type 缺失/非法 → 无关字段（按请求插入顺序取首个）→ 必填字段缺失
    （按对应入口遇到它们的顺序）。字段存在但为 null 时不在此拦截，
    交给对应入口的内容校验；唯 sampled_expectation 的 shots 为 null
    与采样类型矛盾（入口会把 None 解释为精确模式），单独拒绝。
    """
    if not isinstance(request, dict):
        raise ResourceEstimationError(
            "invalid_resource_request", "$", "request must be a JSON object"
        )

    if "type" not in request:
        raise ResourceEstimationError("invalid_resource_request", "type", "missing field type")
    request_type = request["type"]
    if not isinstance(request_type, str) or request_type not in _REQUEST_TYPES:
        raise ResourceEstimationError(
            "invalid_resource_request", "type", f"unsupported request type {request_type!r}"
        )

    allowed = _TYPE_FIELDS[request_type]
    for key in request:
        if not isinstance(key, str) or key not in allowed:
            path = key if isinstance(key, str) else "$"
            raise ResourceEstimationError(
                "invalid_resource_request",
                path,
                f"field {path!r} is not valid for request type {request_type!r}",
            )

    for field in _REQUIRED_FIELDS[request_type]:
        if field not in request:
            raise ResourceEstimationError(
                "invalid_resource_request", field, f"missing field {field}"
            )
    if request_type == "sampled_expectation" and request["shots"] is None:
        raise ResourceEstimationError(
            "invalid_resource_request", "shots", "shots must be a positive integer"
        )
    return request


def estimate_resources(circuit: Any, request: Any, budget: Any = None) -> dict:
    """静态估算运行 ``request`` 所需资源并做预算准入，不执行计算。

    返回 ``{"qubit_count", "type", "representation", "state_elements",
    "state_bytes", "circuit_evaluations", "gate_applications",
    "total_shots", "runtime_supported", "admitted", "exceeded"}``。
    不修改 ``circuit`` / ``request`` / ``budget``，相同输入结果确定，
    输出不含负零。
    """
    # 入口自身的结构校验先于一切：request → budget。
    req = _validate_request(request)
    limits = _validate_budget(budget)
    request_type = req["type"]
    values = req.get("values")
    noise = req.get("noise")

    # 电路规范化与参数绑定对所有类型一致，且在各入口中都最先发生。
    normalized = normalize_circuit(circuit)
    bound = bind_parameters(normalized, {} if values is None else values)
    qubit_count = bound["qubit_count"]
    operation_count = len(bound["operations"])
    r = _count_parameterized_rotations(normalized)

    # 以下内容校验严格沿用各入口的异常类型、code、path 与顺序。
    if request_type == "optimization":
        # optimize：terms 结构 → observables 内容 → noise → config。
        observables, _coefficients = _validate_terms(req["terms"])
        _validate_observables(observables, qubit_count)
        p1, p2 = _validate_noise(noise)
        cfg = _validate_config(req["config"])
        observable_count = len(observables)
        shots_per_evaluation = 0
        circuit_evaluations = (cfg["max_iterations"] + 1) * (1 + 2 * r)
    elif request_type == "sampled_expectation":
        # expectation：observables → noise → shots → seed。
        observables = _validate_observables(req["observables"], qubit_count)
        p1, p2 = _validate_noise(noise)
        shot_count = _validate_shots(req["shots"])
        _validate_seed(req.get("seed"), shot_count)
        observable_count = len(observables)
        shots_per_evaluation = shot_count
        circuit_evaluations = 1
    else:
        # exact_expectation / gradient：observables → noise。
        observables = _validate_observables(req["observables"], qubit_count)
        p1, p2 = _validate_noise(noise)
        observable_count = len(observables)
        shots_per_evaluation = 0
        circuit_evaluations = 1 if request_type == "exact_expectation" else 1 + 2 * r

    noisy = p1 != 0.0 or p2 != 0.0
    if noisy:
        representation = "density_matrix"
        state_elements = 4 ** qubit_count
        runtime_supported = qubit_count <= _MAX_QUBITS_DENSITY_MATRIX
    else:
        representation = "state_vector"
        state_elements = 2 ** qubit_count
        runtime_supported = qubit_count <= _MAX_QUBITS_STATE_VECTOR
    state_bytes = state_elements * _BYTES_PER_ELEMENT

    gate_applications = circuit_evaluations * operation_count
    total_shots = shots_per_evaluation * observable_count

    result: dict[str, Any] = {
        "qubit_count": qubit_count,
        "type": request_type,
        "representation": representation,
        "state_elements": state_elements,
        "state_bytes": state_bytes,
        "circuit_evaluations": circuit_evaluations,
        "gate_applications": gate_applications,
        "total_shots": total_shots,
        "runtime_supported": runtime_supported,
    }

    exceeded: list[str] = []
    if limits is not None:
        usage = {
            "max_state_bytes": state_bytes,
            "max_circuit_evaluations": circuit_evaluations,
            "max_gate_applications": gate_applications,
            "max_total_shots": total_shots,
        }
        for field in _BUDGET_FIELDS:
            if field in limits and usage[field] > limits[field]:
                exceeded.append(field)
    result["exceeded"] = exceeded
    result["admitted"] = runtime_supported and not exceeded
    return result
