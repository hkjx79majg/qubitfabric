"""可序列化的异步参数服务器：多工作方按参数版本提交梯度的确定性合并。

:func:`create_parameter_state` 沿用
:func:`qubitfabric.circuit.normalize_circuit` 与
:func:`qubitfabric.circuit.bind_parameters` 的规范化与绑定语义，从电路
与初始参数值构造参数服务器状态；状态仅含 JSON 原生类型::

    {
        "version": 1,
        "parameters": ["theta"],
        "revision": 0,
        "values": {"theta": 0.3},
        "updates": [
            {"id": "worker-1", "base_revision": 0,
             "gradients": {"theta": 0.5}, "revision": 1,
             "digest": "<sha256>"},
        ],
    }

其中 ``updates`` 是已接收更新的幂等记录，按接收顺序排列，
``digest`` 是更新内容（id、base_revision、gradients）规范 JSON 的
SHA-256。``json.dumps``/``json.loads`` 往返后状态可跨进程传回继续使用，
结果与往返前完全一致。

:func:`apply_parameter_updates` 接收状态、正有限学习率、非负整数
``max_staleness`` 与非空更新数组，按数组顺序处理：

- ``base_revision`` 不大于当前修订号且版本差不超过 ``max_staleness``
  时接收：``value = value - learning_rate * gradient`` 更新全部参数，
  修订号加一并记录内容摘要；
- 版本差超过 ``max_staleness`` 的更新被拒绝（原因 ``stale``），引用
  未来修订的被拒绝（原因 ``future_revision``），均不改变状态；
- 相同 id 且内容相同的更新返回 ``duplicate`` 及首次接收后的修订号，
  不重复更新；相同 id 对应不同内容时整次调用失败。

返回 ``{"state", "results"}``：``state`` 是全新的最终状态，
``results`` 与更新同序，每项含 ``id``、``status``、``reason`` 与
观察到的 ``revision``。全程不修改输入，不使用随机数，不读写文件。

校验顺序：状态（模式、字段、参数集合、数值、幂等记录与摘要完整性）
→ 学习率 → 陈旧度上限 → 更新结构，失败抛 :class:`ParameterServerError`
（``invalid_state`` / ``digest_mismatch`` / ``invalid_learning_rate`` /
``invalid_max_staleness`` / ``invalid_updates`` / ``invalid_update`` /
``idempotency_conflict`` / ``non_finite_result``），异常时不返回部分
状态。创建阶段的电路与绑定错误仍抛
:class:`qubitfabric.circuit.CircuitValidationError` 或
:class:`qubitfabric.circuit.ParameterBindingError`。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .circuit import bind_parameters, normalize_circuit

__all__ = ["ParameterServerError", "create_parameter_state", "apply_parameter_updates"]

_STATE_VERSION = 1
_STATE_FIELDS = ("version", "parameters", "revision", "values", "updates")
_STATE_FIELD_SET = frozenset(_STATE_FIELDS)
_RECORD_FIELDS = ("id", "base_revision", "gradients", "revision", "digest")
_RECORD_FIELD_SET = frozenset(_RECORD_FIELDS)
_UPDATE_FIELDS = ("id", "base_revision", "gradients")
_UPDATE_FIELD_SET = frozenset(_UPDATE_FIELDS)
_HEX_DIGITS = frozenset("0123456789abcdef")


class ParameterServerError(ValueError):
    """参数服务器入口的状态、更新或计算校验失败。

    ``code`` 为稳定的机器可读错误码，``path`` 指向输入中首个出错的
    位置（如 ``state.values.theta``、``learning_rate`` 或
    ``updates[0].gradients``），语义与
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


def _is_number(value: Any) -> bool:
    """JSON 实数（int 或 float），排除布尔值。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canon_float(value: float) -> float:
    """统一浮点形态：抹掉 -0.0。"""
    if value == 0.0:
        return 0.0
    return value


def _finite(value: Any) -> float | None:
    """是有限实数时返回规范 float，否则返回 None。"""
    if not _is_number(value):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return _canon_float(result)


def _canon_json(value: Any) -> str:
    """规范 JSON：键排序、紧凑分隔符、UTF-8（不转义非 ASCII）。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _update_digest(update_id: str, base_revision: int, gradients: dict) -> str:
    """更新内容的稳定摘要：id、base_revision 与规范化的 gradients。"""
    payload = {
        "id": update_id,
        "base_revision": base_revision,
        "gradients": {name: _canon_float(float(gradients[name])) for name in sorted(gradients)},
    }
    return _sha256(_canon_json(payload))


def _invalid_state(path: str, message: str) -> ParameterServerError:
    return ParameterServerError("invalid_state", path, message)


# ---------------------------------------------------------------------------
# 创建入口
# ---------------------------------------------------------------------------

def create_parameter_state(circuit: Any, values: Any) -> dict:
    """从电路与初始参数值构造参数服务器状态，不修改输入。

    电路规范化与参数绑定沿用现有语义，失败抛
    :class:`qubitfabric.circuit.CircuitValidationError` 或
    :class:`qubitfabric.circuit.ParameterBindingError`。返回的状态仅含
    JSON 原生类型：模式版本、参数名、从零开始的修订号、当前有限实数值
    与空的幂等记录。
    """
    normalized = normalize_circuit(circuit)
    bind_parameters(normalized, values)
    parameters = normalized["parameters"]
    return {
        "version": _STATE_VERSION,
        "parameters": list(parameters),
        "revision": 0,
        "values": {name: _canon_float(float(values[name])) for name in parameters},
        "updates": [],
    }


# ---------------------------------------------------------------------------
# 状态校验
# ---------------------------------------------------------------------------

def _validate_parameters(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise _invalid_state("state.parameters", "state parameters must be an array of strings")
    seen: set[str] = set()
    parameters: list[str] = []
    for i, name in enumerate(raw):
        path = f"state.parameters[{i}]"
        if not isinstance(name, str):
            raise _invalid_state(path, f"parameter name at {path} must be a string")
        if name == "":
            raise _invalid_state(path, f"parameter name at {path} must be non-empty")
        if name in seen:
            raise _invalid_state(path, f"duplicate parameter name {name!r} at {path}")
        seen.add(name)
        parameters.append(name)
    return parameters


def _validate_number_map(raw: Any, path: str, parameters: list[str]) -> None:
    """校验 ``{参数名: 有限实数}`` 映射且键集合恰好等于声明参数集。"""
    if not isinstance(raw, dict):
        raise _invalid_state(path, f"{path} must be an object")
    for key in raw:
        if not isinstance(key, str):
            raise _invalid_state(path, f"{path} names must be strings")
    for name, value in raw.items():
        if _finite(value) is None:
            raise _invalid_state(f"{path}.{name}", f"{path}.{name} must be a finite number")
    declared = set(parameters)
    missing = sorted(name for name in declared if name not in raw)
    if missing:
        raise _invalid_state(path, f"{path} is missing parameter {missing[0]!r}")
    unknown = sorted(name for name in raw if name not in declared)
    if unknown:
        raise _invalid_state(path, f"{path} has unknown parameter {unknown[0]!r}")


def _validate_record(raw: Any, index: int, parameters: list[str]) -> None:
    """校验单条幂等记录的结构与摘要完整性。"""
    path = f"state.updates[{index}]"
    if not isinstance(raw, dict):
        raise _invalid_state(path, f"update record at {path} must be an object")
    for key in raw:
        if not isinstance(key, str) or key not in _RECORD_FIELD_SET:
            raise _invalid_state(path, f"unknown update record field at {path}")
    for field in _RECORD_FIELDS:
        if field not in raw:
            raise _invalid_state(path, f"missing update record field {field!r} at {path}")

    update_id = raw["id"]
    if not isinstance(update_id, str) or update_id == "":
        raise _invalid_state(f"{path}.id", f"update record id at {path}.id must be a non-empty string")

    base_revision = raw["base_revision"]
    if not _is_int(base_revision) or base_revision < 0:
        raise _invalid_state(
            f"{path}.base_revision",
            f"update record base_revision at {path}.base_revision must be a non-negative integer",
        )

    _validate_number_map(raw["gradients"], f"{path}.gradients", parameters)

    revision = raw["revision"]
    if not _is_int(revision) or revision < 1:
        raise _invalid_state(
            f"{path}.revision",
            f"update record revision at {path}.revision must be a positive integer",
        )
    if revision != index + 1:
        raise _invalid_state(
            f"{path}.revision",
            f"update record revision at {path}.revision contradicts the receive order",
        )
    if base_revision > revision - 1:
        raise _invalid_state(
            f"{path}.base_revision",
            f"update record base_revision at {path}.base_revision contradicts its revision",
        )

    digest = raw["digest"]
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(ch in _HEX_DIGITS for ch in digest)
    ):
        raise _invalid_state(
            f"{path}.digest",
            f"update record digest at {path}.digest must be 64 lowercase hex characters",
        )
    if _update_digest(update_id, base_revision, raw["gradients"]) != digest:
        raise ParameterServerError(
            "digest_mismatch", f"{path}.digest",
            f"update record digest at {path}.digest does not match its content",
        )


def _validate_state(state: Any) -> tuple[list[str], int, dict, list]:
    """校验状态模式、字段、参数集合、数值与幂等记录，返回深拷贝成分。

    返回 ``(parameters, revision, values, records)``，全部为全新对象；
    结构错误抛 ``invalid_state``，记录摘要与内容不符抛
    ``digest_mismatch``。
    """
    if not isinstance(state, dict):
        raise _invalid_state("state", "state must be an object")
    for key in state:
        if not isinstance(key, str) or key not in _STATE_FIELD_SET:
            raise _invalid_state("state", f"unknown state field {key!r}")
    for field in _STATE_FIELDS:
        if field not in state:
            raise _invalid_state("state", f"missing state field {field!r}")

    version = state["version"]
    if not _is_int(version) or version != _STATE_VERSION:
        raise _invalid_state("state.version", "state version must be 1")

    parameters = _validate_parameters(state["parameters"])

    revision = state["revision"]
    if not _is_int(revision) or revision < 0:
        raise _invalid_state("state.revision", "state revision must be a non-negative integer")

    _validate_number_map(state["values"], "state.values", parameters)

    records = state["updates"]
    if not isinstance(records, list):
        raise _invalid_state("state.updates", "state updates must be an array")
    if len(records) != revision:
        raise _invalid_state(
            "state.updates",
            "state updates length contradicts the revision",
        )
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        _validate_record(record, index, parameters)
        record_id = record["id"]
        if record_id in seen_ids:
            raise _invalid_state(
                f"state.updates[{index}].id",
                f"duplicate update record id {record_id!r} at state.updates[{index}].id",
            )
        seen_ids.add(record_id)

    values = {name: _canon_float(float(state["values"][name])) for name in parameters}
    copied_records = [
        {
            "id": record["id"],
            "base_revision": record["base_revision"],
            "gradients": {
                name: _canon_float(float(record["gradients"][name])) for name in parameters
            },
            "revision": record["revision"],
            "digest": record["digest"],
        }
        for record in records
    ]
    return parameters, revision, values, copied_records


# ---------------------------------------------------------------------------
# 学习率、陈旧度与更新结构校验
# ---------------------------------------------------------------------------

def _validate_learning_rate(learning_rate: Any) -> float:
    """正有限实数学习率。"""
    result = _finite(learning_rate)
    if result is None or result <= 0.0:
        raise ParameterServerError(
            "invalid_learning_rate", "learning_rate",
            "learning_rate must be a positive finite number",
        )
    return result


def _validate_max_staleness(max_staleness: Any) -> int:
    """非负整数陈旧度上限；bool 与非整数统一报 invalid_max_staleness。"""
    if not _is_int(max_staleness) or max_staleness < 0:
        raise ParameterServerError(
            "invalid_max_staleness", "max_staleness",
            "max_staleness must be a non-negative integer",
        )
    return max_staleness


def _validate_updates(updates: Any, parameters: list[str]) -> list[dict]:
    """校验非空更新数组，返回按输入顺序排列的解析结果（全新对象）。"""
    if not isinstance(updates, list) or not updates:
        raise ParameterServerError(
            "invalid_updates", "updates",
            "updates must be a non-empty array",
        )
    parsed: list[dict] = []
    for i, raw in enumerate(updates):
        path = f"updates[{i}]"
        if not isinstance(raw, dict):
            raise ParameterServerError(
                "invalid_update", path, f"update at {path} must be an object",
            )
        for key in raw:
            if not isinstance(key, str) or key not in _UPDATE_FIELD_SET:
                raise ParameterServerError(
                    "invalid_update", path, f"unknown update field at {path}",
                )
        for field in _UPDATE_FIELDS:
            if field not in raw:
                raise ParameterServerError(
                    "invalid_update", path, f"missing update field {field!r} at {path}",
                )

        update_id = raw["id"]
        if not isinstance(update_id, str) or update_id == "":
            raise ParameterServerError(
                "invalid_update", f"{path}.id",
                f"update id at {path}.id must be a non-empty string",
            )

        base_revision = raw["base_revision"]
        if not _is_int(base_revision) or base_revision < 0:
            raise ParameterServerError(
                "invalid_update", f"{path}.base_revision",
                f"update base_revision at {path}.base_revision must be a non-negative integer",
            )

        gradients = raw["gradients"]
        if not isinstance(gradients, dict):
            raise ParameterServerError(
                "invalid_update", f"{path}.gradients",
                f"update gradients at {path}.gradients must be an object",
            )
        for key in gradients:
            if not isinstance(key, str):
                raise ParameterServerError(
                    "invalid_update", f"{path}.gradients",
                    f"update gradient names at {path}.gradients must be strings",
                )
        for name, value in gradients.items():
            if _finite(value) is None:
                raise ParameterServerError(
                    "invalid_update", f"{path}.gradients.{name}",
                    f"update gradient at {path}.gradients.{name} must be a finite number",
                )
        declared = set(parameters)
        missing = sorted(name for name in declared if name not in gradients)
        if missing:
            raise ParameterServerError(
                "invalid_update", f"{path}.gradients",
                f"update gradients at {path}.gradients are missing parameter {missing[0]!r}",
            )
        unknown = sorted(name for name in gradients if name not in declared)
        if unknown:
            raise ParameterServerError(
                "invalid_update", f"{path}.gradients",
                f"update gradients at {path}.gradients have unknown parameter {unknown[0]!r}",
            )

        parsed.append({
            "id": update_id,
            "base_revision": base_revision,
            "gradients": {
                name: _canon_float(float(gradients[name])) for name in parameters
            },
        })
    return parsed


# ---------------------------------------------------------------------------
# 批量更新入口
# ---------------------------------------------------------------------------

def apply_parameter_updates(
    state: Any,
    learning_rate: Any,
    max_staleness: Any,
    updates: Any,
) -> dict:
    """按数组顺序合并一批梯度更新，返回全新状态与同序明细，不修改输入。

    每项更新被接收、拒绝（``stale`` / ``future_revision``）或判为
    ``duplicate``；相同 id 对应不同内容或计算产生非有限数值时整次调用
    抛 :class:`ParameterServerError`，不返回部分状态。返回的状态仅含
    JSON 原生类型，序列化往返后继续使用结果一致。
    """
    parameters, revision, values, records = _validate_state(state)
    rate = _validate_learning_rate(learning_rate)
    staleness = _validate_max_staleness(max_staleness)
    parsed = _validate_updates(updates, parameters)

    record_by_id = {record["id"]: record for record in records}
    results: list[dict] = []

    for i, update in enumerate(parsed):
        update_id = update["id"]
        base_revision = update["base_revision"]
        gradients = update["gradients"]
        digest = _update_digest(update_id, base_revision, gradients)

        existing = record_by_id.get(update_id)
        if existing is not None:
            if existing["digest"] == digest:
                results.append({
                    "id": update_id,
                    "status": "duplicate",
                    "reason": None,
                    "revision": existing["revision"],
                })
                continue
            raise ParameterServerError(
                "idempotency_conflict", f"updates[{i}].id",
                f"update id {update_id!r} at updates[{i}].id was already received with different content",
            )

        if base_revision > revision:
            results.append({
                "id": update_id,
                "status": "rejected",
                "reason": "future_revision",
                "revision": revision,
            })
            continue
        if revision - base_revision > staleness:
            results.append({
                "id": update_id,
                "status": "rejected",
                "reason": "stale",
                "revision": revision,
            })
            continue

        new_values: dict[str, float] = {}
        for name in parameters:
            updated = values[name] - rate * gradients[name]
            if not math.isfinite(updated):
                raise ParameterServerError(
                    "non_finite_result", f"updates[{i}].gradients.{name}",
                    f"updated value for parameter {name!r} at updates[{i}] is not finite",
                )
            new_values[name] = _canon_float(updated)
        values = new_values
        revision += 1

        record = {
            "id": update_id,
            "base_revision": base_revision,
            "gradients": dict(gradients),
            "revision": revision,
            "digest": digest,
        }
        records.append(record)
        record_by_id[update_id] = record
        results.append({
            "id": update_id,
            "status": "accepted",
            "reason": None,
            "revision": revision,
        })

    new_state = {
        "version": _STATE_VERSION,
        "parameters": list(parameters),
        "revision": revision,
        "values": values,
        "updates": records,
    }
    return {"state": new_state, "results": results}
