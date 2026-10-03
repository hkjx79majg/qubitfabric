"""可序列化的异步参数服务器：多工作方梯度提交的确定性合并。

:func:`create_parameter_state` 沿用现有电路规范化与参数绑定语义
（:func:`qubitfabric.circuit.normalize_circuit` /
:func:`qubitfabric.circuit.bind_parameters`），为参数化电路创建初始
参数状态；:func:`apply_parameter_updates` 接收状态、学习率、陈旧度
上限与一批基于不同参数版本计算的梯度更新，按数组顺序确定性地合并，
返回全新的最终状态与同序明细。

状态仅含 JSON 原生类型::

    {
        "version": 1,
        "parameters": ["theta"],
        "revision": 0,
        "values": {"theta": 0.3},
        "updates": [
            {"id": "worker-1", "revision": 1,
             "content": {"base_revision": 0, "gradients": {"theta": 0.1}},
             "digest": "<sha256>"},
        ],
    }

- ``version`` 为模式版本；``revision`` 从零开始，每接受一项更新加一；
- ``values`` 为当前有限实数值；``updates`` 是已接收更新的幂等记录，
  按接收顺序排列，``digest`` 是更新内容（id、base_revision 与规范化
  梯度）经键排序、紧凑分隔符 JSON 序列化后的 SHA-256；
- 状态经 ``json.dumps``/``json.loads`` 往返后可跨进程传回继续使用，
  结果与往返前完全一致。

合并语义（按 updates 数组顺序逐项处理）：

- ``base_revision`` 不大于当前修订号且版本差不超过 ``max_staleness``
  时接受，按 ``value = value - learning_rate * gradient`` 更新全部
  参数，修订号加一并记录内容摘要；
- 版本差超限返回 ``rejected``/``stale``，``base_revision`` 大于当前
  修订号返回 ``rejected``/``future_revision``，均不改变状态；
- 相同 id 与相同内容再次出现返回 ``duplicate`` 及首次接收后的修订号，
  不重复更新；相同 id 对应不同内容时整次调用失败；
- 结果包含全新的最终状态与和更新同序的明细，每项给出 ``id``、
  ``status``、``reason`` 与观察到的 ``revision``。

校验顺序：状态模式与字段、参数集合、数值与幂等记录摘要
（:class:`ParameterServerError`）→ 更新数组结构 → 学习率 → 陈旧度。
格式错误、摘要破坏、幂等冲突或计算产生非有限数值都抛
:class:`ParameterServerError`，``code`` 稳定、``path`` 指向首个错误
位置，异常时不返回部分状态。创建阶段抛出现有
:class:`qubitfabric.circuit.CircuitValidationError` 或
:class:`qubitfabric.circuit.ParameterBindingError`。全程不修改输入，
不读写文件，不使用随机数。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from .circuit import bind_parameters, normalize_circuit

__all__ = ["ParameterServerError", "create_parameter_state", "apply_parameter_updates"]

_STATE_VERSION = 1
_STATE_FIELDS = ("version", "parameters", "revision", "values", "updates")
_STATE_FIELD_SET = frozenset(_STATE_FIELDS)
_RECORD_FIELDS = ("id", "revision", "content", "digest")
_RECORD_FIELD_SET = frozenset(_RECORD_FIELDS)
_CONTENT_FIELDS = ("base_revision", "gradients")
_CONTENT_FIELD_SET = frozenset(_CONTENT_FIELDS)
_UPDATE_FIELDS = ("id", "base_revision", "gradients")
_UPDATE_FIELD_SET = frozenset(_UPDATE_FIELDS)
_HEX_DIGITS = frozenset("0123456789abcdef")


class ParameterServerError(ValueError):
    """参数服务器入口的状态、更新或合并校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_state`` /
    ``digest_mismatch`` / ``invalid_updates`` / ``invalid_update`` /
    ``invalid_learning_rate`` / ``invalid_max_staleness`` /
    ``idempotency_conflict`` / ``non_finite_result``），``path`` 指向
    输入中首个出错的位置（根参数形如 ``"state"``、``"updates"``，内部
    位置形如 ``"state.values.theta"``、``"updates[0].gradients.theta"``），
    语义与 :class:`qubitfabric.circuit.CircuitValidationError` 一致。
    """

    def __init__(self, code: str, path: str, message: str | None = None) -> None:
        self.code = code
        self.path = path
        if message is None:
            message = f"{code} at {path}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# 基础判断
# ---------------------------------------------------------------------------

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


def _canon_json(value: Any) -> str:
    """规范 JSON：键排序、紧凑分隔符、UTF-8（不转义非 ASCII）。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _update_digest(update_id: str, base_revision: int, gradients: dict) -> str:
    """更新内容的稳定摘要：id、base_revision 与梯度映射的规范 JSON。"""
    payload = {"id": update_id, "base_revision": base_revision, "gradients": gradients}
    return hashlib.sha256(_canon_json(payload).encode("utf-8")).hexdigest()


def _is_hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX_DIGITS for ch in value)
    )


def _invalid_state(path: str, message: str) -> ParameterServerError:
    return ParameterServerError("invalid_state", path, message)


def _invalid_update(path: str, message: str) -> ParameterServerError:
    return ParameterServerError("invalid_update", path, message)


# ---------------------------------------------------------------------------
# 状态校验：模式、字段、参数集合、数值与幂等记录摘要
# ---------------------------------------------------------------------------

def _validate_parameters(raw: Any, path: str) -> list[str]:
    if not isinstance(raw, list):
        raise _invalid_state(path, f"{path} must be an array of parameter names")
    parameters: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(raw):
        item_path = f"{path}[{index}]"
        if not isinstance(name, str) or not name:
            raise _invalid_state(item_path, f"parameter name at {item_path} must be a non-empty string")
        if name in seen:
            raise _invalid_state(item_path, f"duplicate parameter name {name!r} at {item_path}")
        seen.add(name)
        parameters.append(name)
    return parameters


def _validate_number_map(raw: Any, path: str, parameters: list[str]) -> None:
    """校验 ``{参数名: 有限实数}`` 映射恰好覆盖参数集合（不修改输入）。"""
    if not isinstance(raw, dict):
        raise _invalid_state(path, f"{path} must be an object mapping parameter names to finite numbers")
    for name, value in raw.items():
        if not isinstance(name, str):
            raise _invalid_state(path, f"{path} keys must be strings")
        if not _is_number(value) or not math.isfinite(float(value)):
            raise _invalid_state(f"{path}.{name}", f"value at {path}.{name} must be a finite number")
    declared = set(parameters)
    missing = sorted(name for name in declared if name not in raw)
    if missing:
        raise _invalid_state(f"{path}.{missing[0]}", f"missing value for parameter {missing[0]!r} at {path}")
    unknown = sorted(name for name in raw if name not in declared)
    if unknown:
        raise _invalid_state(f"{path}.{unknown[0]}", f"unknown parameter {unknown[0]!r} at {path}")


def _validate_record(record: Any, index: int, parameters: list[str], revision: int, floor: int) -> int:
    """校验单条幂等记录并返回其修订号；``floor`` 为上一条记录的修订号。"""
    path = f"state.updates[{index}]"
    if not isinstance(record, dict):
        raise _invalid_state(path, f"update record at {path} must be an object")
    for key in record:
        if not isinstance(key, str) or key not in _RECORD_FIELD_SET:
            raise _invalid_state(path, f"unknown update record field at {path}")
    for field in _RECORD_FIELDS:
        if field not in record:
            raise _invalid_state(f"{path}.{field}", f"missing update record field {path}.{field}")

    record_id = record["id"]
    if not isinstance(record_id, str) or not record_id:
        raise _invalid_state(f"{path}.id", f"update record id at {path}.id must be a non-empty string")

    record_revision = record["revision"]
    if not _is_int(record_revision) or record_revision < 1:
        raise _invalid_state(f"{path}.revision", f"update record revision at {path}.revision must be a positive integer")
    if record_revision <= floor:
        raise _invalid_state(
            f"{path}.revision", f"update record revisions must strictly increase at {path}.revision"
        )
    if record_revision > revision:
        raise _invalid_state(
            f"{path}.revision", f"update record revision at {path}.revision exceeds the state revision"
        )

    content = record["content"]
    content_path = f"{path}.content"
    if not isinstance(content, dict):
        raise _invalid_state(content_path, f"update record content at {content_path} must be an object")
    for key in content:
        if not isinstance(key, str) or key not in _CONTENT_FIELD_SET:
            raise _invalid_state(content_path, f"unknown update content field at {content_path}")
    for field in _CONTENT_FIELDS:
        if field not in content:
            raise _invalid_state(f"{content_path}.{field}", f"missing update content field {content_path}.{field}")

    base_revision = content["base_revision"]
    if not _is_int(base_revision) or base_revision < 0:
        raise _invalid_state(
            f"{content_path}.base_revision",
            f"base_revision at {content_path}.base_revision must be a non-negative integer",
        )
    if base_revision >= record_revision:
        raise _invalid_state(
            f"{content_path}.base_revision",
            f"base_revision at {content_path}.base_revision must be below the record revision",
        )

    _validate_number_map(content["gradients"], f"{content_path}.gradients", parameters)

    digest = record["digest"]
    if not _is_hex_digest(digest):
        raise _invalid_state(f"{path}.digest", f"digest at {path}.digest must be 64 lowercase hex characters")
    if _update_digest(record_id, base_revision, content["gradients"]) != digest:
        raise ParameterServerError(
            "digest_mismatch", f"{path}.digest",
            f"digest at {path}.digest does not match the recorded update content",
        )
    return record_revision


def _validate_state(state: Any) -> tuple[list[str], int, dict, list[dict]]:
    """校验参数状态并返回 ``(parameters, revision, values, records)``。

    结构、字段、参数集合、数值或幂等记录内部一致性错误抛
    ``invalid_state``；记录摘要与内容不符抛 ``digest_mismatch``。
    不修改输入。
    """
    if not isinstance(state, dict):
        raise _invalid_state("state", "state must be an object")
    for key in state:
        if not isinstance(key, str) or key not in _STATE_FIELD_SET:
            path = f"state.{key}" if isinstance(key, str) else "state"
            raise _invalid_state(path, f"unknown state field at {path}")
    for field in _STATE_FIELDS:
        if field not in state:
            raise _invalid_state(f"state.{field}", f"missing state field state.{field}")

    version = state["version"]
    if not _is_int(version) or version != _STATE_VERSION:
        raise _invalid_state("state.version", "state version must be 1")

    parameters = _validate_parameters(state["parameters"], "state.parameters")

    revision = state["revision"]
    if not _is_int(revision) or revision < 0:
        raise _invalid_state("state.revision", "state revision must be a non-negative integer")

    _validate_number_map(state["values"], "state.values", parameters)
    values = {name: _canon_float(float(state["values"][name])) for name in parameters}

    records = state["updates"]
    if not isinstance(records, list):
        raise _invalid_state("state.updates", "state updates must be an array")
    seen_ids: set[str] = set()
    floor = 0
    for index, record in enumerate(records):
        floor = _validate_record(record, index, parameters, revision, floor)
        record_id = record["id"]
        if record_id in seen_ids:
            raise _invalid_state(
                f"state.updates[{index}].id", f"duplicate update id {record_id!r} at state.updates[{index}].id"
            )
        seen_ids.add(record_id)

    return parameters, revision, values, records


# ---------------------------------------------------------------------------
# 更新数组、学习率与陈旧度校验
# ---------------------------------------------------------------------------

def _validate_updates(updates: Any, parameters: list[str]) -> list[tuple[str, int, dict]]:
    """校验更新数组结构，返回 ``(id, base_revision, 规范化梯度)`` 列表。"""
    if not isinstance(updates, list) or not updates:
        raise ParameterServerError(
            "invalid_updates", "updates", "updates must be a non-empty array"
        )
    declared = set(parameters)
    parsed: list[tuple[str, int, dict]] = []
    for index, update in enumerate(updates):
        path = f"updates[{index}]"
        if not isinstance(update, dict):
            raise _invalid_update(path, f"update at {path} must be an object")
        for key in update:
            if not isinstance(key, str) or key not in _UPDATE_FIELD_SET:
                key_path = f"{path}.{key}" if isinstance(key, str) else path
                raise _invalid_update(key_path, f"unknown update field at {key_path}")
        for field in _UPDATE_FIELDS:
            if field not in update:
                raise _invalid_update(f"{path}.{field}", f"missing update field {path}.{field}")

        update_id = update["id"]
        if not isinstance(update_id, str) or not update_id:
            raise _invalid_update(f"{path}.id", f"update id at {path}.id must be a non-empty string")

        base_revision = update["base_revision"]
        if not _is_int(base_revision) or base_revision < 0:
            raise _invalid_update(
                f"{path}.base_revision",
                f"base_revision at {path}.base_revision must be a non-negative integer",
            )

        gradients = update["gradients"]
        gradients_path = f"{path}.gradients"
        if not isinstance(gradients, dict):
            raise _invalid_update(
                gradients_path, f"gradients at {gradients_path} must be an object mapping names to finite numbers"
            )
        for name, value in gradients.items():
            if not isinstance(name, str):
                raise _invalid_update(gradients_path, f"gradient names at {gradients_path} must be strings")
            if not _is_number(value) or not math.isfinite(float(value)):
                raise _invalid_update(
                    f"{gradients_path}.{name}", f"gradient at {gradients_path}.{name} must be a finite number"
                )
        missing = sorted(name for name in declared if name not in gradients)
        if missing:
            raise _invalid_update(
                f"{gradients_path}.{missing[0]}",
                f"missing gradient for parameter {missing[0]!r} at {gradients_path}",
            )
        unknown = sorted(name for name in gradients if name not in declared)
        if unknown:
            raise _invalid_update(
                f"{gradients_path}.{unknown[0]}",
                f"unknown gradient parameter {unknown[0]!r} at {gradients_path}",
            )

        canonical = {name: _canon_float(float(gradients[name])) for name in parameters}
        parsed.append((update_id, base_revision, canonical))
    return parsed


def _validate_learning_rate(learning_rate: Any) -> float:
    """正有限学习率；bool、非数值、非正或非有限统一报 invalid_learning_rate。"""
    if not _is_number(learning_rate):
        raise ParameterServerError(
            "invalid_learning_rate", "learning_rate", "learning_rate must be a positive finite number"
        )
    result = float(learning_rate)
    if not math.isfinite(result) or result <= 0.0:
        raise ParameterServerError(
            "invalid_learning_rate", "learning_rate", "learning_rate must be a positive finite number"
        )
    return result


def _validate_max_staleness(max_staleness: Any) -> int:
    """非负整数陈旧度上限；bool、非整数或负数统一报 invalid_max_staleness。"""
    if not _is_int(max_staleness) or max_staleness < 0:
        raise ParameterServerError(
            "invalid_max_staleness", "max_staleness", "max_staleness must be a non-negative integer"
        )
    return max_staleness


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def create_parameter_state(circuit: Any, values: Any = None) -> dict:
    """为参数化电路创建初始参数状态，不修改输入。

    电路规范化与参数绑定沿用
    :func:`qubitfabric.circuit.normalize_circuit` 与
    :func:`qubitfabric.circuit.bind_parameters` 的语义，失败抛出现有
    :class:`qubitfabric.circuit.CircuitValidationError` 或
    :class:`qubitfabric.circuit.ParameterBindingError`。返回的状态仅含
    JSON 原生类型：模式版本 ``version``、参数名 ``parameters``、从零
    开始的修订号 ``revision``、当前有限实数值 ``values`` 与为空的幂等
    记录 ``updates``。
    """
    normalized = normalize_circuit(circuit)
    resolved = {} if values is None else values
    bind_parameters(normalized, resolved)
    parameters = normalized["parameters"]
    return {
        "version": _STATE_VERSION,
        "parameters": list(parameters),
        "revision": 0,
        "values": {name: _canon_float(float(resolved[name])) for name in parameters},
        "updates": [],
    }


def apply_parameter_updates(
    state: Any,
    learning_rate: Any,
    max_staleness: Any,
    updates: Any,
) -> dict:
    """按数组顺序确定性地合并一批梯度更新，不修改输入。

    返回 ``{"state", "details"}``：``state`` 为全新的最终状态，
    ``details`` 与更新同序，每项为 ``{"id", "status", "reason",
    "revision"}``；``status`` 取 ``accepted``/``rejected``/``duplicate``，
    ``reason`` 在拒绝时为 ``stale``/``future_revision``，其余为 None，
    ``revision`` 为该项观察到的修订号（接受后为新增一的修订号，重复时
    为首次接收后的修订号，拒绝时为当前修订号）。状态、更新结构、学习率
    或陈旧度校验失败，以及幂等冲突、计算产生非有限数值，都抛
    :class:`ParameterServerError`，异常时不返回部分状态。
    """
    parameters, revision, values, records = _validate_state(state)
    parsed = _validate_updates(updates, parameters)
    rate = _validate_learning_rate(learning_rate)
    staleness_limit = _validate_max_staleness(max_staleness)

    new_values = dict(values)
    new_revision = revision
    new_records = copy.deepcopy(records)
    by_id = {record["id"]: record for record in new_records}
    details: list[dict] = []

    for index, (update_id, base_revision, gradients) in enumerate(parsed):
        digest = _update_digest(update_id, base_revision, gradients)
        existing = by_id.get(update_id)
        if existing is not None:
            if existing["digest"] == digest:
                details.append({
                    "id": update_id,
                    "status": "duplicate",
                    "reason": None,
                    "revision": existing["revision"],
                })
                continue
            raise ParameterServerError(
                "idempotency_conflict", f"updates[{index}].id",
                f"update id {update_id!r} at updates[{index}].id was already received with different content",
            )

        if base_revision > new_revision:
            details.append({
                "id": update_id,
                "status": "rejected",
                "reason": "future_revision",
                "revision": new_revision,
            })
            continue
        if new_revision - base_revision > staleness_limit:
            details.append({
                "id": update_id,
                "status": "rejected",
                "reason": "stale",
                "revision": new_revision,
            })
            continue

        for name in parameters:
            updated = new_values[name] - rate * gradients[name]
            if not math.isfinite(updated):
                raise ParameterServerError(
                    "non_finite_result", f"updates[{index}].gradients.{name}",
                    f"applying update at updates[{index}] makes parameter {name!r} non-finite",
                )
            new_values[name] = _canon_float(updated)
        new_revision += 1
        record = {
            "id": update_id,
            "revision": new_revision,
            "content": {"base_revision": base_revision, "gradients": dict(gradients)},
            "digest": digest,
        }
        new_records.append(record)
        by_id[update_id] = record
        details.append({
            "id": update_id,
            "status": "accepted",
            "reason": None,
            "revision": new_revision,
        })

    new_state = {
        "version": _STATE_VERSION,
        "parameters": list(parameters),
        "revision": new_revision,
        "values": new_values,
        "updates": new_records,
    }
    return {"state": new_state, "details": details}
