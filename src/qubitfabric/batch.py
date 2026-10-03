"""同一电路上的批量 Pauli 期望值估计。

调用方针对同一份电路提交多个相互独立的作业，每个作业携带自己的
``observables`` 以及可选的 ``values``/``shots``/``seed``/``noise``，语义与
:func:`qubitfabric.simulate.estimate_expectation` 完全一致。可选的
``max_concurrency`` 限制同时执行的作业数，省略时为 1（串行）。

入口先按现有语义规范化公共电路（失败抛
:class:`qubitfabric.circuit.CircuitValidationError`），随后校验
``max_concurrency`` 与 ``jobs`` 结构（失败抛 :class:`BatchExecutionError`），
再开始执行作业。单个作业的参数绑定或仿真校验失败只影响该作业本身：
对应项标记为 ``failed`` 并携带原异常的类名、``code`` 与 ``path``，
其余作业继续执行。结果始终按 ``jobs`` 输入顺序排列；相同输入在不同
合法并行度下得到内容完全相同的结果。所有输入对象均不被修改。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .circuit import ParameterBindingError, normalize_circuit
from .simulate import SimulationError, estimate_expectation

__all__ = ["BatchExecutionError", "expectation_batch"]

_JOB_FIELDS = frozenset(("id", "observables", "values", "shots", "seed", "noise"))


class BatchExecutionError(ValueError):
    """批量期望值入口的结构校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_batch``/``invalid_job``/
    ``duplicate_job_id``/``invalid_concurrency``），``path`` 指向输入中
    出错的位置，语义与
    :class:`qubitfabric.circuit.CircuitValidationError` 一致。
    """

    def __init__(self, code: str, path: str, message: str | None = None) -> None:
        self.code = code
        self.path = path
        if message is None:
            message = f"{code} at {path}"
        super().__init__(message)


def _validate_max_concurrency(value: Any) -> int:
    """校验并行度：省略为 1，显式值必须是排除 bool 的正整数。"""
    if value is None:
        return 1
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BatchExecutionError(
            "invalid_concurrency", "max_concurrency",
            "max_concurrency must be a positive integer",
        )
    return value


def _validate_jobs(jobs: Any) -> None:
    """校验 jobs 结构：非空数组，每项含唯一非空字符串 id 与 observables。"""
    if not isinstance(jobs, list) or not jobs:
        raise BatchExecutionError(
            "invalid_batch", "jobs",
            "jobs must be a non-empty array of job objects",
        )
    seen: set[str] = set()
    for i, job in enumerate(jobs):
        base = f"jobs[{i}]"
        if not isinstance(job, dict):
            raise BatchExecutionError(
                "invalid_job", base, f"job at {base} must be an object",
            )
        if "id" not in job or not isinstance(job["id"], str) or not job["id"]:
            raise BatchExecutionError(
                "invalid_job", f"{base}.id",
                f"job at {base} must have a non-empty string id",
            )
        if "observables" not in job:
            raise BatchExecutionError(
                "invalid_job", f"{base}.observables",
                f"job at {base} is missing observables",
            )
        unknown = sorted(
            (key for key in job if key not in _JOB_FIELDS),
            key=str,
        )
        if unknown:
            key = unknown[0]
            path = f"{base}.{key}" if isinstance(key, str) else base
            raise BatchExecutionError(
                "invalid_job", path, f"unknown job field at {path}",
            )
        if job["id"] in seen:
            raise BatchExecutionError(
                "duplicate_job_id", f"{base}.id",
                f"duplicate job id {job['id']!r} at {base}.id",
            )
        seen.add(job["id"])


def _run_job(normalized: dict, job: dict) -> dict:
    """执行单个作业；绑定或仿真校验失败转化为 failed 项。"""
    try:
        result = estimate_expectation(
            normalized,
            job["observables"],
            values=job.get("values"),
            shots=job.get("shots"),
            seed=job.get("seed"),
            noise=job.get("noise"),
        )
    except (ParameterBindingError, SimulationError) as exc:
        return {
            "id": job["id"],
            "status": "failed",
            "result": {},
            "error": {
                "type": type(exc).__name__,
                "code": exc.code,
                "path": exc.path,
            },
        }
    return {"id": job["id"], "status": "succeeded", "result": result, "error": None}


def expectation_batch(circuit: Any, jobs: Any, max_concurrency: Any = None) -> dict:
    """在同一电路上批量估计 Pauli 期望值，不修改输入。

    返回 ``{"results", "summary"}``：results 按 ``jobs`` 输入顺序排列，
    成功项为 ``{"id", "status": "succeeded", "result", "error": None}``，
    其中 ``result`` 与单独调用单次入口完全一致；失败项为
    ``{"id", "status": "failed", "result": {}, "error"}``，``error`` 给出
    原异常类名、稳定的 ``code`` 与 ``path``。``summary`` 为
    ``{"total", "succeeded", "failed"}`` 三个整数，与逐项状态一致。
    """
    normalized = normalize_circuit(circuit)
    workers = _validate_max_concurrency(max_concurrency)
    _validate_jobs(jobs)

    if workers == 1:
        outcomes = [_run_job(normalized, job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(lambda job: _run_job(normalized, job), jobs))

    succeeded = sum(1 for item in outcomes if item["status"] == "succeeded")
    summary = {
        "total": len(outcomes),
        "succeeded": succeeded,
        "failed": len(outcomes) - succeeded,
    }
    return {"results": outcomes, "summary": summary}
