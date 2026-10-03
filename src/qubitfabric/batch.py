"""同一电路上多个独立 Pauli 期望值作业的批量执行。

:func:`run_batch` 先按现有语义规范化公共电路（失败继续抛
:class:`qubitfabric.circuit.CircuitValidationError`），随后校验
``max_concurrency`` 与 ``jobs`` 结构，再开始作业。作业在线程池中
受限并行执行，实际同时执行的作业数绝不超过 ``max_concurrency``
（省略为 1）；单个作业的参数绑定或仿真失败只影响该项，其余作业
继续。相同输入在任意合法并行度下产生内容相同的结果，且与逐项
单独调用 :func:`qubitfabric.simulate.estimate_expectation` 完全
一致；不修改任何输入对象，不引入缓存、落盘或第三方依赖。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .circuit import (
    CircuitValidationError,
    ParameterBindingError,
    normalize_circuit,
)
from .simulate import SimulationError, estimate_expectation

__all__ = ["BatchExecutionError", "run_batch"]

_JOB_FIELDS = ("id", "observables", "values", "shots", "seed", "noise")
_JOB_FIELD_SET = frozenset(_JOB_FIELDS)


class BatchExecutionError(ValueError):
    """批量入口的请求级结构校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_batch`` /
    ``invalid_job`` / ``duplicate_job_id`` / ``invalid_concurrency``），
    ``path`` 指向输入中首个出错位置，语义与
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


def _validate_concurrency(max_concurrency: Any) -> int:
    """省略为 1；显式值必须是排除 bool 的正整数。"""
    if max_concurrency is None:
        return 1
    if not _is_int(max_concurrency) or max_concurrency < 1:
        raise BatchExecutionError(
            "invalid_concurrency", "max_concurrency",
            "max_concurrency must be a positive integer",
        )
    return max_concurrency


def _validate_jobs(jobs: Any) -> list[dict]:
    """校验 jobs 结构并返回原对象构成的列表（只读使用，不修改）。

    非数组或空数组报 ``invalid_batch``/``jobs``；作业不是对象、id
    缺失/非字符串/空串、observables 缺失或存在范围外字段报
    ``invalid_job``，path 指向按作业顺序遇到的首个问题；重复 id 报
    ``duplicate_job_id``，指向后出现的 ``jobs[i].id``。
    """
    if not isinstance(jobs, list) or not jobs:
        raise BatchExecutionError(
            "invalid_batch", "jobs",
            "jobs must be a non-empty array",
        )

    seen: set[str] = set()
    valid: list[dict] = []
    for i, job in enumerate(jobs):
        path = f"jobs[{i}]"
        if not isinstance(job, dict):
            raise BatchExecutionError(
                "invalid_job", path,
                f"job at {path} must be an object",
            )
        for key in job:
            if not isinstance(key, str) or key not in _JOB_FIELD_SET:
                key_path = f"{path}.{key}" if isinstance(key, str) else path
                raise BatchExecutionError(
                    "invalid_job", key_path,
                    f"unknown job field at {key_path}",
                )
        if "id" not in job:
            raise BatchExecutionError(
                "invalid_job", f"{path}.id",
                f"missing field {path}.id",
            )
        job_id = job["id"]
        if not isinstance(job_id, str) or not job_id:
            raise BatchExecutionError(
                "invalid_job", f"{path}.id",
                f"field {path}.id must be a non-empty string",
            )
        if "observables" not in job:
            raise BatchExecutionError(
                "invalid_job", f"{path}.observables",
                f"missing field {path}.observables",
            )
        if job_id in seen:
            raise BatchExecutionError(
                "duplicate_job_id", f"{path}.id",
                f"duplicate job id {job_id!r} at {path}.id",
            )
        seen.add(job_id)
        valid.append(job)
    return valid


def _run_one(normalized: dict, job: dict) -> dict:
    """执行单个作业，沿用单次 expectation 的全部校验与仿真语义。

    作业各可选字段缺省时显式传 None，与单次入口省略参数同义；
    estimate_expectation 内部重新绑定参数并自行规范化，不会修改
    规范化电路或作业对象。
    """
    return estimate_expectation(
        normalized,
        job["observables"],
        values=job.get("values", None),
        shots=job.get("shots", None),
        seed=job.get("seed", None),
        noise=job.get("noise", None),
    )


def run_batch(circuit: Any, jobs: Any, max_concurrency: Any = None) -> dict:
    """在同一规范化电路上批量执行多个独立期望值作业。

    返回 ``{"results", "summary"}``：results 始终按 jobs 输入顺序
    排列，成功项为 ``{"id", "status": "succeeded", "result",
    "error": None}``，失败项为 ``{"id", "status": "failed",
    "result": None, "error"}``，其中 error 为
    ``{"type", "code", "path", "message"}``；summary 给出
    ``total``/``succeeded``/``failed``。请求级结构错误抛
    :class:`BatchExecutionError`，公共电路校验失败抛
    :class:`CircuitValidationError`，二者都发生在任何作业开始之前。
    不修改输入。
    """
    normalized = normalize_circuit(circuit)
    concurrency = _validate_concurrency(max_concurrency)
    valid_jobs = _validate_jobs(jobs)

    results: list[dict | None] = [None] * len(valid_jobs)
    succeeded = 0
    failed = 0

    def execute(index: int, job: dict) -> tuple[int, dict]:
        try:
            result = _run_one(normalized, job)
        except (CircuitValidationError, ParameterBindingError, SimulationError) as exc:
            return index, {
                "id": job["id"],
                "status": "failed",
                "result": None,
                "error": {
                    "type": type(exc).__name__,
                    "code": exc.code,
                    "path": exc.path,
                    "message": str(exc),
                },
            }
        return index, {
            "id": job["id"],
            "status": "succeeded",
            "result": result,
            "error": None,
        }

    # 有界线程池：max_workers 即同时执行作业数的硬上界；作业按输入
    # 顺序提交，但结果用提交下标归位，输出次序与完成先后无关。
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute, i, job) for i, job in enumerate(valid_jobs)]
        for future in futures:
            index, item = future.result()
            results[index] = item
            if item["status"] == "succeeded":
                succeeded += 1
            else:
                failed += 1

    return {
        "results": results,
        "summary": {"total": len(valid_jobs), "succeeded": succeeded, "failed": failed},
    }
