"""可暂停、跨进程续算的确定性变分优化。

:func:`optimize_resumable` 接收与
:func:`qubitfabric.optimize.optimize_circuit` 相同的基础输入
（circuit、terms、values、config 与可选 noise），另接收正整数
``step_budget`` 更新预算与可选 ``checkpoint``：

- 首次调用（无检查点）产生与 ``optimize`` 相同的 iteration 0 评估，
  之后每次调用至多执行 ``step_budget`` 次参数更新；
- 梯度范数达到 ``tolerance`` 或累计更新达到 ``config.max_iterations``
  时结束，返回 ``{"status": "completed", "result", "checkpoint": None,
  "progress"}``，``result`` 与相同输入直接调用 ``optimize`` 完全一致；
- 否则暂停，返回 ``{"status": "paused", "result": None, "checkpoint",
  "progress"}``，``progress`` 含 ``iterations``、``parameters``、
  ``values``、``objective``、``gradient_norm`` 与从 iteration 0 起的
  完整 ``history``。

检查点仅由 JSON 原生类型组成，``json.dumps``/``json.loads`` 往返后可
跨进程续算；任意分段的最终数值、参数顺序与 ``history`` 都与单次
``optimize`` 一致。全程不修改输入，不读写文件。

校验顺序：基础优化输入沿用 ``optimize`` 的顺序与异常，其后依次校验
``step_budget`` 与 ``checkpoint``，两者失败抛 :class:`RuntimeStateError`
（``invalid_step_budget`` / ``invalid_checkpoint`` /
``checkpoint_mismatch``）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .optimize import (
    _advance,
    _canon_float,
    _finalize,
    _finite,
    _initial_state,
    _is_int,
    _prepare,
)

__all__ = ["RuntimeStateError", "optimize_resumable"]

_CHECKPOINT_VERSION = 1
_CHECKPOINT_FIELDS = (
    "version",
    "fingerprint",
    "iterations",
    "values",
    "objective",
    "gradient_norm",
    "gradients",
    "first_moment",
    "second_moment",
    "history",
)
_CHECKPOINT_FIELD_SET = frozenset(_CHECKPOINT_FIELDS)
_HISTORY_FIELDS = ("iteration", "values", "objective", "gradient_norm")
_HISTORY_FIELD_SET = frozenset(_HISTORY_FIELDS)
_STATE_MAP_FIELDS = ("values", "gradients", "first_moment", "second_moment")


class RuntimeStateError(ValueError):
    """可恢复优化的运行时状态（更新预算 / 检查点）校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_step_budget``、
    ``invalid_checkpoint``、``checkpoint_mismatch``），``path`` 指向
    输入中出错的位置（``step_budget`` 或 ``checkpoint``），语义与
    :class:`qubitfabric.circuit.CircuitValidationError` 一致。
    """

    def __init__(self, code: str, path: str, message: str | None = None) -> None:
        self.code = code
        self.path = path
        if message is None:
            message = f"{code} at {path}"
        super().__init__(message)


def _invalid(message: str) -> RuntimeStateError:
    return RuntimeStateError("invalid_checkpoint", "checkpoint", message)


# ---------------------------------------------------------------------------
# step_budget
# ---------------------------------------------------------------------------

def _validate_step_budget(step_budget: Any) -> int:
    """正整数更新预算；bool、非整数与小于 1 统一报 invalid_step_budget。"""
    if not _is_int(step_budget) or step_budget < 1:
        raise RuntimeStateError(
            "invalid_step_budget", "step_budget",
            "step_budget must be a positive integer",
        )
    return step_budget


# ---------------------------------------------------------------------------
# 输入指纹：检查点只能由相同的 circuit、terms、初始 values、config 和
# noise 产生，否则报 checkpoint_mismatch。
# ---------------------------------------------------------------------------

def _fingerprint(
    normalized: dict,
    initial: dict,
    observables: list[str],
    coefficients: list[float],
    noise_probs: tuple[float, float],
    cfg: dict,
    parameters: list[str],
) -> str:
    payload = {
        "kind": "qubitfabric.optimize_resumable",
        "version": _CHECKPOINT_VERSION,
        "circuit": normalized,
        "terms": {"observables": observables, "coefficients": coefficients},
        "values": {name: _canon_float(float(initial[name])) for name in parameters},
        "config": cfg,
        "noise": [noise_probs[0], noise_probs[1]],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# checkpoint 校验与恢复
# ---------------------------------------------------------------------------

def _validate_number_map(value: Any, field: str) -> None:
    """校验 ``{参数名: 有限实数}`` 映射的结构（不比对具体参数集）。"""
    if not isinstance(value, dict):
        raise _invalid(f"checkpoint field {field} must be an object")
    for key, item in value.items():
        if not isinstance(key, str):
            raise _invalid(f"checkpoint field {field} names must be strings")
        if _finite(item) is None:
            raise _invalid(f"checkpoint field {field} must map names to finite numbers")


def _validate_checkpoint(
    checkpoint: Any,
    cfg: dict,
    parameters: list[str],
    fingerprint: str,
) -> None:
    """校验检查点结构、内部一致性与输入指纹。

    结构错误（不是对象、版本不支持、字段缺失或未知、类型错误、非有限
    数、状态矛盾）抛 ``invalid_checkpoint``；结构合法但指纹与当前输入
    不符抛 ``checkpoint_mismatch``。
    """
    if not isinstance(checkpoint, dict):
        raise _invalid("checkpoint must be an object")
    for key in checkpoint:
        if not isinstance(key, str) or key not in _CHECKPOINT_FIELD_SET:
            raise _invalid(f"unknown checkpoint field {key!r}")
    for field in _CHECKPOINT_FIELDS:
        if field not in checkpoint:
            raise _invalid(f"missing checkpoint field {field!r}")

    version = checkpoint["version"]
    if not _is_int(version) or version != _CHECKPOINT_VERSION:
        raise _invalid(f"unsupported checkpoint version {version!r}")

    if not isinstance(checkpoint["fingerprint"], str):
        raise _invalid("checkpoint fingerprint must be a string")

    iterations = checkpoint["iterations"]
    if not _is_int(iterations) or iterations < 0:
        raise _invalid("checkpoint iterations must be a non-negative integer")

    for field in _STATE_MAP_FIELDS:
        _validate_number_map(checkpoint[field], field)

    if _finite(checkpoint["objective"]) is None:
        raise _invalid("checkpoint objective must be a finite number")
    norm = _finite(checkpoint["gradient_norm"])
    if norm is None or norm < 0.0:
        raise _invalid("checkpoint gradient_norm must be a non-negative finite number")

    names = set(checkpoint["values"])
    for field in ("gradients", "first_moment", "second_moment"):
        if set(checkpoint[field]) != names:
            raise _invalid(f"checkpoint field {field} keys must match values keys")

    history = checkpoint["history"]
    if not isinstance(history, list) or not history:
        raise _invalid("checkpoint history must be a non-empty array")
    if len(history) != iterations + 1:
        raise _invalid("checkpoint history length contradicts iterations")
    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            raise _invalid("checkpoint history entries must be objects")
        for key in entry:
            if not isinstance(key, str) or key not in _HISTORY_FIELD_SET:
                raise _invalid(f"unknown checkpoint history field {key!r}")
        for field in _HISTORY_FIELDS:
            if field not in entry:
                raise _invalid(f"missing checkpoint history field {field!r}")
        if not _is_int(entry["iteration"]) or entry["iteration"] != index:
            raise _invalid("checkpoint history iterations must number from 0")
        _validate_number_map(entry["values"], "history values")
        if set(entry["values"]) != names:
            raise _invalid("checkpoint history values keys must match values keys")
        if _finite(entry["objective"]) is None:
            raise _invalid("checkpoint history objective must be a finite number")
        entry_norm = _finite(entry["gradient_norm"])
        if entry_norm is None or entry_norm < 0.0:
            raise _invalid("checkpoint history gradient_norm must be a non-negative finite number")

    last = history[-1]
    if (
        last["values"] != checkpoint["values"]
        or last["objective"] != checkpoint["objective"]
        or last["gradient_norm"] != checkpoint["gradient_norm"]
    ):
        raise _invalid("checkpoint state contradicts its last history entry")

    if checkpoint["fingerprint"] != fingerprint:
        raise RuntimeStateError(
            "checkpoint_mismatch", "checkpoint",
            "checkpoint was not produced by the same circuit, terms, values, config and noise",
        )

    # 指纹一致后仍可能被人为构造：暂停状态不可能已收敛或已达更新上限，
    # 参数集也必须与电路声明一致，否则为状态矛盾。
    if names != set(parameters):
        raise _invalid("checkpoint parameter set contradicts the circuit")
    if norm <= cfg["tolerance"]:
        raise _invalid("checkpoint state has already converged")
    if iterations >= cfg["max_iterations"]:
        raise _invalid("checkpoint state has already reached max_iterations")
    for entry in history:
        if entry["gradient_norm"] <= cfg["tolerance"]:
            raise _invalid("checkpoint history contains a converged state")


def _restore(checkpoint: dict, parameters: list[str]) -> dict:
    """把校验过的检查点还原为循环状态；全部拷贝，不共享输入对象。"""
    history = [
        {
            "iteration": entry["iteration"],
            "values": {name: float(entry["values"][name]) for name in parameters},
            "objective": float(entry["objective"]),
            "gradient_norm": float(entry["gradient_norm"]),
        }
        for entry in checkpoint["history"]
    ]
    return {
        "iterations": checkpoint["iterations"],
        "current": {name: float(checkpoint["values"][name]) for name in parameters},
        "gradients": {name: float(checkpoint["gradients"][name]) for name in parameters},
        "objective": float(checkpoint["objective"]),
        "norm": float(checkpoint["gradient_norm"]),
        "history": history,
        "first_moment": {name: float(checkpoint["first_moment"][name]) for name in parameters},
        "second_moment": {name: float(checkpoint["second_moment"][name]) for name in parameters},
    }


def _checkpoint_of(state: dict, parameters: list[str], fingerprint: str) -> dict:
    """把循环状态导出为仅含 JSON 原生类型的检查点。"""
    return {
        "version": _CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "iterations": state["iterations"],
        "values": {name: _canon_float(state["current"][name]) for name in parameters},
        "objective": _canon_float(state["objective"]),
        "gradient_norm": _canon_float(state["norm"]),
        "gradients": {name: _canon_float(state["gradients"][name]) for name in parameters},
        "first_moment": {name: _canon_float(state["first_moment"][name]) for name in parameters},
        "second_moment": {name: _canon_float(state["second_moment"][name]) for name in parameters},
        "history": state["history"],
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def optimize_resumable(
    circuit: Any,
    terms: Any,
    values: Any,
    config: Any,
    noise: Any = None,
    step_budget: Any = None,
    checkpoint: Any = None,
) -> dict:
    """按更新预算推进的确定性变分优化，不修改输入，不使用随机数。

    返回 ``{"status", "result", "checkpoint", "progress"}``：未结束时
    ``status`` 为 ``"paused"``、``result`` 为 None、``checkpoint`` 可
    序列化后传回续算；结束时 ``status`` 为 ``"completed"``、
    ``checkpoint`` 为 None、``result`` 与相同输入直接调用
    :func:`qubitfabric.optimize.optimize_circuit` 完全一致。
    """
    normalized, initial, observables, coefficients, noise_probs, cfg, parameters, evaluate = (
        _prepare(circuit, terms, values, config, noise)
    )
    budget = _validate_step_budget(step_budget)
    fingerprint = _fingerprint(
        normalized, initial, observables, coefficients, noise_probs, cfg, parameters,
    )

    if checkpoint is None:
        state = _initial_state(parameters, initial, evaluate)
    else:
        _validate_checkpoint(checkpoint, cfg, parameters, fingerprint)
        state = _restore(checkpoint, parameters)

    _advance(state, cfg, parameters, evaluate, budget)
    done = state["norm"] <= cfg["tolerance"] or state["iterations"] >= cfg["max_iterations"]

    progress = {
        "iterations": state["iterations"],
        "parameters": parameters,
        "values": {name: _canon_float(state["current"][name]) for name in parameters},
        "objective": _canon_float(state["objective"]),
        "gradient_norm": _canon_float(state["norm"]),
        "history": state["history"],
    }
    if done:
        return {
            "status": "completed",
            "result": _finalize(state, cfg, parameters),
            "checkpoint": None,
            "progress": progress,
        }
    return {
        "status": "paused",
        "result": None,
        "checkpoint": _checkpoint_of(state, parameters, fingerprint),
        "progress": progress,
    }
