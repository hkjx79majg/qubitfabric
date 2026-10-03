"""批量期望值作业的离线卸载规划：只规划，不执行仿真。

:func:`plan_batch_offload` 先按现有语义规范化公共电路（失败继续抛
:class:`qubitfabric.circuit.CircuitValidationError`），随后校验
``backends`` 与 ``jobs`` 的请求级结构，再按 jobs 顺序逐作业规划。
规划复用 ``exact_expectation``（``shots`` 为 None）或
``sampled_expectation`` 的校验与资源口径，但不执行任何仿真。

每个后端声明支持的表示（``state_vector`` / ``density_matrix``）、
四项资源预算上限（与 :mod:`qubitfabric.resources` 的预算键同名）与
正整数 ``slots``。合法作业只能分配给表示受支持、各项需求未超预算
且仍有空闲 slot 的后端；候选中选择 ``assigned_count/slots`` 最小者
（整数交叉乘法比较，相等时取输入靠前者），分配成功后占用一个 slot。

语义无效的作业返回 ``rejected`` 且不阻断其他作业；没有任何合格后端
时返回 ``no_eligible_backend``，并按后端输入顺序列出每个后端落选的
原因（``unsupported_representation`` / ``no_slot`` / 超限预算键）。

请求级失败（backends/jobs 结构）抛 :class:`OffloadPlanningError` 或
:class:`qubitfabric.batch.BatchExecutionError`，不返回部分计划。
输出仅含 JSON 原生类型，不修改任何输入对象。
"""

from __future__ import annotations

from typing import Any

from .batch import _validate_jobs
from .circuit import (
    CircuitValidationError,
    ParameterBindingError,
    bind_parameters,
    normalize_circuit,
)
from .resources import _BUDGET_KEYS
from .simulate import (
    SimulationError,
    _validate_noise,
    _validate_observables,
    _validate_seed,
    _validate_shots,
)

__all__ = ["OffloadPlanningError", "plan_batch_offload"]

_BYTES_PER_ELEMENT = 16
_REPRESENTATIONS = ("state_vector", "density_matrix")
_REPRESENTATION_SET = frozenset(_REPRESENTATIONS)
_BACKEND_FIELDS = ("id", "representations", "slots") + _BUDGET_KEYS
_BACKEND_FIELD_SET = frozenset(_BACKEND_FIELDS)


class OffloadPlanningError(ValueError):
    """卸载规划入口的请求级校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_backends`` /
    ``invalid_backend`` / ``duplicate_backend_id``），``path`` 指向输入
    中首个出错位置，语义与
    :class:`qubitfabric.circuit.CircuitValidationError` 一致。
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
# backends 请求级校验
# ---------------------------------------------------------------------------

def _validate_backends(backends: Any) -> list[dict]:
    """校验 backends 结构，返回后端对象列表（只读使用，不修改）。

    非数组或空数组报 ``invalid_backends``/``backends``；后端不是对象、
    含未知字段、id 缺失或非非空字符串、representations 缺失/非非空数组/
    含非法值/重复、预算键缺失或非排除 bool 的非负整数、slots 缺失或非
    正整数，报 ``invalid_backend``，path 指向按后端顺序遇到的首个问题；
    重复 id 报 ``duplicate_backend_id``，指向后出现的 ``backends[i].id``。
    """
    if not isinstance(backends, list) or not backends:
        raise OffloadPlanningError(
            "invalid_backends", "backends",
            "backends must be a non-empty array",
        )

    seen: set[str] = set()
    valid: list[dict] = []
    for i, backend in enumerate(backends):
        path = f"backends[{i}]"
        if not isinstance(backend, dict):
            raise OffloadPlanningError(
                "invalid_backend", path,
                f"backend at {path} must be an object",
            )
        for key in backend:
            if not isinstance(key, str) or key not in _BACKEND_FIELD_SET:
                key_path = f"{path}.{key}" if isinstance(key, str) else path
                raise OffloadPlanningError(
                    "invalid_backend", key_path,
                    f"unknown backend field at {key_path}",
                )

        backend_id = backend.get("id")
        if not isinstance(backend_id, str) or not backend_id:
            raise OffloadPlanningError(
                "invalid_backend", f"{path}.id",
                f"field {path}.id must be a non-empty string",
            )

        representations = backend.get("representations")
        rep_path = f"{path}.representations"
        if not isinstance(representations, list) or not representations:
            raise OffloadPlanningError(
                "invalid_backend", rep_path,
                f"field {rep_path} must be a non-empty array of representations",
            )
        seen_reps: set[str] = set()
        for j, representation in enumerate(representations):
            item_path = f"{rep_path}[{j}]"
            if not isinstance(representation, str) or representation not in _REPRESENTATION_SET:
                raise OffloadPlanningError(
                    "invalid_backend", item_path,
                    f"representation at {item_path} must be one of "
                    "state_vector, density_matrix",
                )
            if representation in seen_reps:
                raise OffloadPlanningError(
                    "invalid_backend", item_path,
                    f"duplicate representation {representation!r} at {item_path}",
                )
            seen_reps.add(representation)

        for key in _BUDGET_KEYS:
            value = backend.get(key)
            if not _is_int(value) or value < 0:
                raise OffloadPlanningError(
                    "invalid_backend", f"{path}.{key}",
                    f"field {path}.{key} must be a non-negative integer",
                )

        slots = backend.get("slots")
        if not _is_int(slots) or slots < 1:
            raise OffloadPlanningError(
                "invalid_backend", f"{path}.slots",
                f"field {path}.slots must be a positive integer",
            )

        if backend_id in seen:
            raise OffloadPlanningError(
                "duplicate_backend_id", f"{path}.id",
                f"duplicate backend id {backend_id!r} at {path}.id",
            )
        seen.add(backend_id)
        valid.append(backend)
    return valid


# ---------------------------------------------------------------------------
# 单作业校验与资源口径
# ---------------------------------------------------------------------------

def _validate_job(normalized: dict, job: dict) -> tuple[str, int, int, int, int, int]:
    """沿用 exact/sampled expectation 的校验顺序，返回资源需求六元组。

    依次为参数绑定、observables、噪声、shots、seed；shots 为 None 走
    exact_expectation 口径（禁止 seed），否则走 sampled_expectation 口径。
    返回 ``(representation, state_bytes, evaluations, gate_applications,
    total_shots, qubit_count)``。
    """
    bound = bind_parameters(normalized, {} if job.get("values") is None else job["values"])
    qubit_count = bound["qubit_count"]
    observables = _validate_observables(job["observables"], qubit_count)
    p1, p2 = _validate_noise(job.get("noise"))
    shots = _validate_shots(job.get("shots"))
    _validate_seed(job.get("seed"), shots)

    if p1 != 0.0 or p2 != 0.0:
        representation = "density_matrix"
        state_elements = 1 << (2 * qubit_count)
    else:
        representation = "state_vector"
        state_elements = 1 << qubit_count
    state_bytes = state_elements * _BYTES_PER_ELEMENT
    evaluations = 1
    gate_applications = len(normalized["operations"])
    total_shots = shots * len(observables) if shots is not None else 0
    return representation, state_bytes, evaluations, gate_applications, total_shots, qubit_count


def _requirements(
    representation: str,
    state_bytes: int,
    evaluations: int,
    gate_applications: int,
    total_shots: int,
    qubit_count: int,
) -> dict:
    """构造仅含 JSON 原生类型的需求画像（键与资源估计口径一致）。"""
    return {
        "qubit_count": qubit_count,
        "representation": representation,
        "state_elements": state_bytes // _BYTES_PER_ELEMENT,
        "state_bytes": state_bytes,
        "circuit_evaluations": evaluations,
        "gate_applications": gate_applications,
        "total_shots": total_shots,
    }


# ---------------------------------------------------------------------------
# 规划
# ---------------------------------------------------------------------------

def plan_batch_offload(circuit: Any, jobs: Any, backends: Any) -> dict:
    """把批量期望值作业规划到后端，只规划不执行仿真，不修改输入。

    返回 ``{"results", "backends", "summary"}``：results 按 jobs 输入
    顺序排列；成功项为 ``{"id", "status": "assigned", "assigned": true,
    "backend_id", "requirements", "reasons": null, "validation_error":
    null}``；语义无效项 status 为 ``rejected`` 并带 ``validation_error``
    （``{"type", "code", "path", "message"}``）；无合格后端时 status 为
    ``no_eligible_backend``，``reasons`` 按后端顺序给出每个后端的落选
    原因。``backends`` 按输入顺序给出 ``{"id", "slots",
    "assigned_count"}``；summary 给出 ``total``/``assigned``/``rejected``。
    """
    # 校验顺序固定为电路、backends、jobs 结构；任一请求级失败不产出计划。
    normalized = normalize_circuit(circuit)
    valid_backends = _validate_backends(backends)
    valid_jobs = _validate_jobs(jobs)

    backend_ids = [backend["id"] for backend in valid_backends]
    backend_reps = [list(backend["representations"]) for backend in valid_backends]
    backend_slots = [backend["slots"] for backend in valid_backends]
    backend_limits = [
        {budget: backend[budget] for budget in _BUDGET_KEYS}
        for backend in valid_backends
    ]
    assigned_counts = [0] * len(valid_backends)

    results: list[dict] = []
    assigned_total = 0
    rejected_total = 0

    for job in valid_jobs:
        try:
            representation, state_bytes, evaluations, gate_applications, total_shots, qubit_count = (
                _validate_job(normalized, job)
            )
        except (CircuitValidationError, ParameterBindingError, SimulationError) as exc:
            rejected_total += 1
            results.append({
                "id": job["id"],
                "status": "rejected",
                "assigned": False,
                "backend_id": None,
                "requirements": None,
                "reasons": None,
                "validation_error": {
                    "type": type(exc).__name__,
                    "code": exc.code,
                    "path": exc.path,
                    "message": str(exc),
                },
            })
            continue

        needs = {
            "max_state_bytes": state_bytes,
            "max_circuit_evaluations": evaluations,
            "max_gate_applications": gate_applications,
            "max_total_shots": total_shots,
        }

        chosen = -1
        reasons: list[dict] = []
        # 选择最小 assigned_count/slots：交叉乘法 a/b < c/d 等价于 a*d < c*b，
        # 严格比较保证比值相等时保留输入顺序更靠前的候选。
        best_num = 0
        best_den = 1  # 首个候选以 chosen == -1 直接接纳，后续按严格小于更新。
        for index in range(len(valid_backends)):
            backend_reasons: list[str] = []
            if representation not in backend_reps[index]:
                backend_reasons.append("unsupported_representation")
            if assigned_counts[index] >= backend_slots[index]:
                backend_reasons.append("no_slot")
            for budget in _BUDGET_KEYS:
                if needs[budget] > backend_limits[index][budget]:
                    backend_reasons.append(budget)

            if backend_reasons:
                reasons.append({
                    "backend_id": backend_ids[index],
                    "reasons": backend_reasons,
                })
                continue

            num = assigned_counts[index]
            den = backend_slots[index]
            if chosen == -1 or num * best_den < best_num * den:
                chosen = index
                best_num = num
                best_den = den

        if chosen == -1:
            rejected_total += 1
            results.append({
                "id": job["id"],
                "status": "no_eligible_backend",
                "assigned": False,
                "backend_id": None,
                "requirements": None,
                "reasons": reasons,
                "validation_error": None,
            })
            continue

        assigned_counts[chosen] += 1
        assigned_total += 1
        results.append({
            "id": job["id"],
            "status": "assigned",
            "assigned": True,
            "backend_id": valid_backends[chosen]["id"],
            "requirements": _requirements(
                representation, state_bytes, evaluations,
                gate_applications, total_shots, qubit_count,
            ),
            "reasons": None,
            "validation_error": None,
        })

    return {
        "results": results,
        "backends": [
            {
                "id": backend["id"],
                "slots": backend["slots"],
                "assigned_count": assigned_counts[index],
            }
            for index, backend in enumerate(valid_backends)
        ],
        "summary": {
            "total": len(valid_jobs),
            "assigned": assigned_total,
            "rejected": rejected_total,
        },
    }
