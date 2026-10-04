"""批量期望值作业的后端卸载规划（只规划、不执行仿真）。

:func:`plan_batch_offload` 先按现有语义规范化公共电路（失败继续抛
:class:`qubitfabric.circuit.CircuitValidationError`），随后校验
``backends`` 与 ``jobs`` 的请求级结构（分别抛
:class:`OffloadPlanningError` 与
:class:`qubitfabric.batch.BatchExecutionError`），再按 jobs 顺序逐项
处理：每个合法作业按 ``shots`` 是否为 None 沿用
``exact_expectation`` / ``sampled_expectation`` 的校验与资源口径
（与 :func:`qubitfabric.resources.estimate_resources` 完全一致，不
执行任何仿真），把表示受支持、四项资源预算未超限且仍有空位的后端
作为候选，选 ``assigned_count / slots`` 最小者（整数交叉乘法比较，
相等时取输入靠前者）并占用一个 slot。语义无效的作业只拒绝该项，
不阻断其他作业。不修改任何输入，输出仅含 JSON 原生类型。

:func:`diagnose_batch_offload` 沿用完全相同的请求级校验与逐作业
规划过程（因此 ``plan`` 与 :func:`plan_batch_offload` 同输入的结果
逐值一致），但只诊断、不产生任何额外副作用，并额外返回按 jobs
顺序排列的 ``diagnostics``：每个后端在处理该作业前的已占槽位数、
slot 总量、是否候选、不适用原因与四项预算的 required/limit/exceeded
快照，使调用方可以复核每次选择。
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

__all__ = ["OffloadPlanningError", "plan_batch_offload", "diagnose_batch_offload"]

_BYTES_PER_ELEMENT = 16
_MAX_QUBITS_STATE_VECTOR = 20
_MAX_QUBITS_DENSITY_MATRIX = 10

_BACKEND_FIELDS = (
    "id",
    "representations",
    "max_state_bytes",
    "max_circuit_evaluations",
    "max_gate_applications",
    "max_total_shots",
    "slots",
)
_BACKEND_FIELD_SET = frozenset(_BACKEND_FIELDS)
_REPRESENTATIONS = ("state_vector", "density_matrix")
_REPRESENTATION_SET = frozenset(_REPRESENTATIONS)


class OffloadPlanningError(ValueError):
    """卸载规划入口的请求级校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_backends`` /
    ``invalid_backend`` / ``duplicate_backend_id``），``path`` 指向
    输入中首个出错位置，语义与
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


def _validate_backends(backends: Any) -> list[dict]:
    """校验后端数组结构，返回后端对象构成的列表（只读使用，不修改）。

    非数组或空数组报 ``invalid_backends``/``backends``；后端不是对象、
    含未知字段、id 缺失/非字符串/空串、representations 缺失/非数组/为空/
    含非法或重复表示、预算缺失或不是排除 bool 的非负整数、slots 缺失或
    不是排除 bool 的正整数，均报 ``invalid_backend``，path 指向按后端
    顺序遇到的首个问题；重复 id 报 ``duplicate_backend_id``，指向后
    出现的 ``backends[i].id``。
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
        if "id" not in backend:
            raise OffloadPlanningError(
                "invalid_backend", f"{path}.id",
                f"missing field {path}.id",
            )
        backend_id = backend["id"]
        if not isinstance(backend_id, str) or not backend_id:
            raise OffloadPlanningError(
                "invalid_backend", f"{path}.id",
                f"field {path}.id must be a non-empty string",
            )
        if "representations" not in backend:
            raise OffloadPlanningError(
                "invalid_backend", f"{path}.representations",
                f"missing field {path}.representations",
            )
        reps = backend["representations"]
        reps_path = f"{path}.representations"
        if not isinstance(reps, list) or not reps:
            raise OffloadPlanningError(
                "invalid_backend", reps_path,
                f"field {reps_path} must be a non-empty array",
            )
        for j, rep in enumerate(reps):
            rep_path = f"{reps_path}[{j}]"
            if not isinstance(rep, str) or rep not in _REPRESENTATION_SET:
                raise OffloadPlanningError(
                    "invalid_backend", rep_path,
                    f"representation at {rep_path} must be state_vector or density_matrix",
                )
        if len(set(reps)) != len(reps):
            for j, rep in enumerate(reps):
                if rep in reps[:j]:
                    raise OffloadPlanningError(
                        "invalid_backend", f"{reps_path}[{j}]",
                        f"duplicate representation {rep!r} at {reps_path}[{j}]",
                    )
        for key in _BUDGET_KEYS:
            if key not in backend:
                raise OffloadPlanningError(
                    "invalid_backend", f"{path}.{key}",
                    f"missing field {path}.{key}",
                )
            value = backend[key]
            if not _is_int(value) or value < 0:
                raise OffloadPlanningError(
                    "invalid_backend", f"{path}.{key}",
                    f"field {path}.{key} must be a non-negative integer",
                )
        if "slots" not in backend:
            raise OffloadPlanningError(
                "invalid_backend", f"{path}.slots",
                f"missing field {path}.slots",
            )
        slots = backend["slots"]
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


def _job_requirements(normalized: dict, job: dict) -> dict:
    """按 exact/sampled expectation 口径校验单个作业并估算资源需求。

    校验顺序与 :func:`qubitfabric.simulate.estimate_expectation` 一致：
    参数绑定 → observables → noise → shots → seed；资源口径与
    :func:`qubitfabric.resources.estimate_resources` 一致。任何语义
    错误原样抛出（CircuitValidationError/ParameterBindingError/
    SimulationError），由调用方转写为 rejected 项。
    """
    bound = bind_parameters(normalized, {} if job.get("values") is None else job["values"])
    qubit_count = bound["qubit_count"]
    _validate_observables(job["observables"], qubit_count)
    p1, p2 = _validate_noise(job.get("noise"))
    shots = _validate_shots(job.get("shots"))
    _validate_seed(job.get("seed"), shots)

    noisy = p1 != 0.0 or p2 != 0.0
    max_qubits = _MAX_QUBITS_DENSITY_MATRIX if noisy else _MAX_QUBITS_STATE_VECTOR
    if qubit_count > max_qubits:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {max_qubits}-qubit simulation limit",
        )

    if noisy:
        representation = "density_matrix"
        state_elements = 1 << (2 * qubit_count)
    else:
        representation = "state_vector"
        state_elements = 1 << qubit_count
    state_bytes = state_elements * _BYTES_PER_ELEMENT

    evaluations = 1
    gate_applications = len(normalized["operations"])
    total_shots = 0 if shots is None else shots * len(job["observables"])

    return {
        "representation": representation,
        "state_bytes": state_bytes,
        "circuit_evaluations": evaluations,
        "gate_applications": gate_applications,
        "total_shots": total_shots,
    }


_REQUIREMENT_KEYS = {
    "max_state_bytes": "state_bytes",
    "max_circuit_evaluations": "circuit_evaluations",
    "max_gate_applications": "gate_applications",
    "max_total_shots": "total_shots",
}


def _backend_reasons(backend: dict, assigned_count: int, requirements: dict) -> list[str]:
    """给出单个后端对当前作业不适用的原因列表。

    原因取值 ``unsupported_representation`` / ``no_slot`` /
    超限预算键（``max_state_bytes`` 等）；同一后端可同时报多个原因，
    顺序为表示、slot、各预算键（预算键按规范顺序）。
    """
    reasons: list[str] = []
    if requirements["representation"] not in backend["representations"]:
        reasons.append("unsupported_representation")
    if assigned_count >= backend["slots"]:
        reasons.append("no_slot")
    for budget_key in _BUDGET_KEYS:
        if requirements[_REQUIREMENT_KEYS[budget_key]] > backend[budget_key]:
            reasons.append(budget_key)
    return reasons


def _prepare_offload_inputs(circuit: Any, jobs: Any, backends: Any) -> tuple[dict, list[dict], list[dict]]:
    """两个公共入口共用的请求级校验：电路 → backends → jobs 结构。"""
    normalized = normalize_circuit(circuit)
    valid_backends = _validate_backends(backends)
    valid_jobs = _validate_jobs(jobs)
    return normalized, valid_jobs, valid_backends


def _validation_error(exc: Exception) -> dict:
    """把单项语义异常转写为 rejected 项携带的稳定错误对象。"""
    return {
        "type": type(exc).__name__,
        "code": exc.code,
        "path": exc.path,
        "message": str(exc),
    }


def _public_requirements(requirements: dict) -> dict:
    """复制一份仅含公开资源键的需求对象。"""
    return {
        "representation": requirements["representation"],
        "state_bytes": requirements["state_bytes"],
        "circuit_evaluations": requirements["circuit_evaluations"],
        "gate_applications": requirements["gate_applications"],
        "total_shots": requirements["total_shots"],
    }


def _candidate_budgets(backend: dict, requirements: dict) -> dict:
    """四个预算键各自的 required/limit/exceeded 快照（按规范键序）。"""
    budgets: dict[str, dict] = {}
    for budget_key in _BUDGET_KEYS:
        required = requirements[_REQUIREMENT_KEYS[budget_key]]
        limit = backend[budget_key]
        budgets[budget_key] = {
            "required": required,
            "limit": limit,
            "exceeded": required > limit,
        }
    return budgets


def _run_offload(
    normalized: dict,
    valid_jobs: list[dict],
    valid_backends: list[dict],
    diagnose: bool,
) -> dict:
    """逐作业规划核心；``diagnose`` 为真时额外记录候选审计快照。

    无论是否诊断，选择与计数逻辑完全一致，因此同输入的规划结果逐值
    相同。候选快照在选择循环内、占用 slot 之前记录，反映处理该作业
    之前的逐后端状态。
    """
    # 内部计数载体独立于输入对象，绝不写回入参后端。
    assigned_counts = [0] * len(valid_backends)

    results: list[dict] = []
    diagnostics: list[dict] = []
    assigned = 0
    rejected = 0

    for job in valid_jobs:
        try:
            requirements = _job_requirements(normalized, job)
        except (CircuitValidationError, ParameterBindingError, SimulationError) as exc:
            rejected += 1
            error = _validation_error(exc)
            results.append({
                "id": job["id"],
                "status": "rejected",
                "requirements": None,
                "validation_error": error,
            })
            if diagnose:
                diagnostics.append({
                    "id": job["id"],
                    "status": "rejected",
                    "requirements": None,
                    "selected_backend_id": None,
                    "candidates": [],
                    "validation_error": error,
                })
            continue

        req_public = _public_requirements(requirements)

        chosen = -1
        # 候选比例 assigned_count/slots 的最小者；交叉乘法 a/b < c/d
        # 等价于 a*d < c*b（slots 为正整数），相等时保留输入靠前者。
        best_num = 0
        best_den = 1
        all_reasons: list[list[str]] = []
        candidates: list[dict] = []
        for i, backend in enumerate(valid_backends):
            assigned_before = assigned_counts[i]
            reasons = _backend_reasons(backend, assigned_before, requirements)
            all_reasons.append(reasons)
            if diagnose:
                candidates.append({
                    "backend_id": backend["id"],
                    "assigned_before": assigned_before,
                    "slots": backend["slots"],
                    "eligible": not reasons,
                    "reasons": reasons,
                    "budgets": _candidate_budgets(backend, requirements),
                })
            if reasons:
                continue
            num = assigned_before
            den = backend["slots"]
            if chosen == -1 or num * best_den < best_num * den:
                chosen = i
                best_num = num
                best_den = den

        if chosen == -1:
            rejected += 1
            results.append({
                "id": job["id"],
                "status": "no_eligible_backend",
                "requirements": None,
                "reasons": all_reasons,
            })
            if diagnose:
                diagnostics.append({
                    "id": job["id"],
                    "status": "no_eligible_backend",
                    "requirements": req_public,
                    "selected_backend_id": None,
                    "candidates": candidates,
                })
            continue

        assigned_counts[chosen] += 1
        assigned += 1
        backend_id = valid_backends[chosen]["id"]
        results.append({
            "id": job["id"],
            "status": "assigned",
            "backend_id": backend_id,
            "requirements": req_public,
        })
        if diagnose:
            diagnostics.append({
                "id": job["id"],
                "status": "assigned",
                "requirements": req_public,
                "selected_backend_id": backend_id,
                "candidates": candidates,
            })

    summary = {
        "backend_assignments": [
            {"backend_id": backend["id"], "assigned": count}
            for backend, count in zip(valid_backends, assigned_counts)
        ],
        "total": len(valid_jobs),
        "assigned": assigned,
        "rejected": rejected,
    }
    plan = {"results": results, "summary": summary}
    if diagnose:
        return {"plan": plan, "diagnostics": diagnostics}
    return plan


def plan_batch_offload(circuit: Any, jobs: Any, backends: Any) -> dict:
    """把批量期望值作业规划分配到后端，不执行任何仿真，不修改输入。

    返回 ``{"results", "summary"}``：results 按 jobs 输入顺序排列，
    成功项为 ``{"id", "status": "assigned", "backend_id",
    "requirements"}``，语义无效项为 ``{"id", "status": "rejected",
    "requirements": None, "validation_error"}``，无候选后端时为
    ``{"id", "status": "no_eligible_backend", "requirements": None,
    "reasons"}``；summary 给出每个后端的分配数 ``backend_assignments``
    （按后端输入顺序）以及 ``total``/``assigned``/``rejected``。
    请求级失败（电路/backends/jobs 结构）直接抛异常，不返回部分计划。
    """
    normalized, valid_jobs, valid_backends = _prepare_offload_inputs(circuit, jobs, backends)
    return _run_offload(normalized, valid_jobs, valid_backends, diagnose=False)


def diagnose_batch_offload(circuit: Any, jobs: Any, backends: Any) -> dict:
    """只诊断、不执行仿真：规划批量作业并审计逐作业的后端选择过程。

    输入、请求级校验顺序与异常（电路
    :class:`qubitfabric.circuit.CircuitValidationError`、backends
    :class:`OffloadPlanningError`、jobs 结构
    :class:`qubitfabric.batch.BatchExecutionError`）与
    :func:`plan_batch_offload` 完全相同，请求级失败不返回部分结果。

    返回 ``{"plan", "diagnostics"}``：``plan`` 与同输入直接调用
    :func:`plan_batch_offload` 的结果逐值一致（字段、顺序、值）；
    ``diagnostics`` 按 jobs 输入顺序与 plan results 一一对应。每个
    语义合法项（``assigned`` / ``no_eligible_backend``）含 ``id``、
    ``status``、``requirements``、``selected_backend_id`` 与
    ``candidates``：成功分配时 ``selected_backend_id`` 为后端 id，
    否则为 None；``candidates`` 按 backends 顺序列出**全部**后端，
    每项含 ``backend_id``、``assigned_before``（处理该作业前该后端
    已占槽位数）、``slots``、``eligible``（仅当 reasons 为空时为
    true）、``reasons``（沿用 ``unsupported_representation``、
    ``no_slot`` 与四个预算键的既有顺序）与 ``budgets``（四个预算键
    各给 ``required``/``limit``/``exceeded``，按
    ``max_state_bytes`` / ``max_circuit_evaluations`` /
    ``max_gate_applications`` / ``max_total_shots`` 顺序）。
    ``no_eligible_backend`` 项保留已计算的 ``requirements`` 与全部
    候选原因。语义无效项使用 ``rejected``：``requirements`` 与
    ``selected_backend_id`` 为 None、``candidates`` 为空，并原样
    携带与规划结果一致的 ``validation_error``；单项失败不阻断后续
    作业。候选快照反映逐作业推进时（占用 slot 之前）的状态，调用
    方可按 assigned_before/slots 的最小负载比例与输入顺序复核选择。
    输出仅含 JSON 原生类型，不修改输入，相同输入结果完全相同。
    """
    normalized, valid_jobs, valid_backends = _prepare_offload_inputs(circuit, jobs, backends)
    return _run_offload(normalized, valid_jobs, valid_backends, diagnose=True)
