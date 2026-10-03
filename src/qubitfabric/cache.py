"""带 LRU 缓存与跨进程稳定请求身份的 Pauli 期望值估计。

:func:`cached_expectation` 接收与
:func:`qubitfabric.simulate.estimate_expectation` 相同的基础输入
（circuit、observables、values、shots、seed、noise），另接收缓存快照
``cache`` 与容量上限 ``max_entries``（省略为 128），返回
``{"result", "cache_hit", "request_id", "cache"}``：

- ``result`` 与相同输入直接调用 ``estimate_expectation`` 完全一致；
- ``request_id`` 是请求身份的 SHA-256（64 位小写十六进制），身份由
  完整绑定后的规范电路、保留顺序和重复项的 observables、解析后的
  shots、seed（采样省略 seed 时沿用现有默认种子 0）及补齐默认值的
  noise 组成，按键排序、紧凑分隔符、UTF-8 的 JSON 序列化后计算；
  省略默认值或等价数字形式不改变身份；
- ``cache`` 为 ``{"version": 1, "entries"}``，每项仅含
  ``request_id``、``result``、``result_digest``（result 规范 JSON 的
  SHA-256），按最近使用优先排列，仅含 JSON 原生类型，``json.dumps``/
  ``json.loads`` 往返后可跨进程传回复用。

命中时核对摘要、返回独立副本并把条目移到首位；未命中时按现有语义
计算并插入首位，超出 ``max_entries`` 淘汰末项。全程不修改输入，
不读写文件，不引入第三方依赖。

校验顺序：基础输入沿用 ``estimate_expectation`` 的顺序、异常类型与
量子位限制；其后依次校验 ``max_entries`` 与 ``cache``，失败抛
:class:`CacheStateError`（``invalid_cache_capacity`` /
``invalid_cache``）。结构合法但不含当前身份的快照只是普通未命中。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

from .circuit import bind_parameters
from .simulate import (
    _MAX_QUBITS,
    _MAX_QUBITS_NOISY,
    SimulationError,
    _validate_noise,
    _validate_observables,
    _validate_seed,
    _validate_shots,
    estimate_expectation,
)

__all__ = ["CacheStateError", "cached_expectation"]

_CACHE_VERSION = 1
_DEFAULT_MAX_ENTRIES = 128
_CACHE_FIELDS = ("version", "entries")
_CACHE_FIELD_SET = frozenset(_CACHE_FIELDS)
_ENTRY_FIELDS = ("request_id", "result", "result_digest")
_ENTRY_FIELD_SET = frozenset(_ENTRY_FIELDS)
_HEX_DIGITS = frozenset("0123456789abcdef")


class CacheStateError(ValueError):
    """缓存入口的容量上限或缓存快照校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_cache_capacity`` /
    ``invalid_cache``），``path`` 指向输入中出错的位置
    （``max_entries`` 或 ``cache``），语义与
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


def _invalid(message: str) -> CacheStateError:
    return CacheStateError("invalid_cache", "cache", message)


def _canon_json(value: Any) -> str:
    """规范 JSON：键排序、紧凑分隔符、UTF-8（不转义非 ASCII）。"""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 基础输入：与 estimate_expectation 相同的校验顺序、异常与量子位限制
# ---------------------------------------------------------------------------

def _resolve_base(
    circuit: Any,
    observables: Any,
    values: Any,
    shots: Any,
    seed: Any,
    noise: Any,
) -> tuple[dict, list[str], int | None, int | None, float, float]:
    """校验基础请求并返回解析后的身份成分。

    返回 ``(bound, pauli_strings, shot_count, resolved_seed, p1, p2)``；
    校验顺序、异常类型与量子位限制和 ``estimate_expectation`` 一致。
    """
    bound = bind_parameters(circuit, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    pauli_strings = _validate_observables(observables, qubit_count)
    p1, p2 = _validate_noise(noise)
    shot_count = _validate_shots(shots)
    resolved_seed = _validate_seed(seed, shot_count)

    noisy = p1 != 0.0 or p2 != 0.0
    max_qubits = _MAX_QUBITS_NOISY if noisy else _MAX_QUBITS
    if qubit_count > max_qubits:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {max_qubits}-qubit simulation limit",
        )
    return bound, pauli_strings, shot_count, resolved_seed, p1, p2


def _request_id(
    bound: dict,
    pauli_strings: list[str],
    shot_count: int | None,
    resolved_seed: int | None,
    p1: float,
    p2: float,
) -> str:
    """请求身份：绑定后的规范电路、observables、shots、seed 与补齐默认值的 noise。"""
    payload = {
        "circuit": bound,
        "observables": pauli_strings,
        "shots": shot_count,
        "seed": resolved_seed,
        "noise": {
            "single_qubit_depolarizing": p1,
            "two_qubit_depolarizing": p2,
        },
    }
    return _sha256(_canon_json(payload))


# ---------------------------------------------------------------------------
# max_entries 与 cache 快照校验
# ---------------------------------------------------------------------------

def _validate_max_entries(max_entries: Any) -> int:
    """省略为 128；显式值必须是排除 bool 的正整数。"""
    if max_entries is None:
        return _DEFAULT_MAX_ENTRIES
    if not _is_int(max_entries) or max_entries < 1:
        raise CacheStateError(
            "invalid_cache_capacity", "max_entries",
            "max_entries must be a positive integer",
        )
    return max_entries


def _is_hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX_DIGITS for ch in value)
    )


def _is_json_native(value: Any) -> bool:
    """仅含 JSON 原生类型（有限数；对象键为字符串）。"""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_native(item)
            for key, item in value.items()
        )
    return False


def _validate_cache(cache: Any) -> list[dict]:
    """校验缓存快照结构、内部一致性并返回条目列表（不修改输入）。

    非对象、版本错误、字段缺失或未知、entries 或条目结构错误、
    request_id 重复或格式错误、摘要格式错误或与 result 不符统一抛
    ``invalid_cache``；结构合法即视为可复用快照。
    """
    if not isinstance(cache, dict):
        raise _invalid("cache must be an object")
    for key in cache:
        if not isinstance(key, str) or key not in _CACHE_FIELD_SET:
            raise _invalid(f"unknown cache field {key!r}")
    for field in _CACHE_FIELDS:
        if field not in cache:
            raise _invalid(f"missing cache field {field!r}")

    version = cache["version"]
    if not _is_int(version) or version != _CACHE_VERSION:
        raise _invalid("cache version must be 1")

    entries = cache["entries"]
    if not isinstance(entries, list):
        raise _invalid("cache entries must be an array")

    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise _invalid(f"cache entry at entries[{index}] must be an object")
        for key in entry:
            if not isinstance(key, str) or key not in _ENTRY_FIELD_SET:
                raise _invalid(f"unknown cache entry field {key!r} at entries[{index}]")
        for field in _ENTRY_FIELDS:
            if field not in entry:
                raise _invalid(f"missing cache entry field {field!r} at entries[{index}]")

        request_id = entry["request_id"]
        if not _is_hex_digest(request_id):
            raise _invalid(f"request_id at entries[{index}] must be 64 lowercase hex characters")
        if request_id in seen:
            raise _invalid(f"duplicate request_id at entries[{index}]")
        seen.add(request_id)

        digest = entry["result_digest"]
        if not _is_hex_digest(digest):
            raise _invalid(f"result_digest at entries[{index}] must be 64 lowercase hex characters")

        result = entry["result"]
        if not _is_json_native(result):
            raise _invalid(f"result at entries[{index}] must contain only JSON native types")
        if _sha256(_canon_json(result)) != digest:
            raise _invalid(f"result_digest at entries[{index}] does not match result")

    return entries


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def cached_expectation(
    circuit: Any,
    observables: Any,
    values: Any = None,
    shots: Any = None,
    seed: Any = None,
    noise: Any = None,
    cache: Any = None,
    max_entries: Any = None,
) -> dict:
    """带 LRU 缓存的期望值估计，不修改输入，缓存可 JSON 往返跨进程复用。

    基础输入的含义、校验顺序、异常类型与量子位限制和
    :func:`qubitfabric.simulate.estimate_expectation` 完全一致；
    ``max_entries`` 省略为 128，显式值必须是排除 bool 的正整数，否则抛
    :class:`CacheStateError`（``invalid_cache_capacity``）；``cache``
    快照结构或内部一致性错误抛 :class:`CacheStateError`
    （``invalid_cache``）。返回 ``{"result", "cache_hit", "request_id",
    "cache"}``，``result`` 与直接调用 ``estimate_expectation`` 一致。
    """
    bound, pauli_strings, shot_count, resolved_seed, p1, p2 = _resolve_base(
        circuit, observables, values, shots, seed, noise,
    )
    capacity = _validate_max_entries(max_entries)
    entries = _validate_cache(cache)

    request_id = _request_id(bound, pauli_strings, shot_count, resolved_seed, p1, p2)

    for index, entry in enumerate(entries):
        if entry["request_id"] == request_id:
            ordered = [entry, *entries[:index], *entries[index + 1:]]
            return {
                "result": copy.deepcopy(entry["result"]),
                "cache_hit": True,
                "request_id": request_id,
                "cache": {"version": _CACHE_VERSION, "entries": copy.deepcopy(ordered)},
            }

    result = estimate_expectation(
        circuit, observables, values=values, shots=shots, seed=seed, noise=noise,
    )
    entry = {
        "request_id": request_id,
        "result": copy.deepcopy(result),
        "result_digest": _sha256(_canon_json(result)),
    }
    new_entries = [entry, *copy.deepcopy(entries)]
    del new_entries[capacity:]
    return {
        "result": result,
        "cache_hit": False,
        "request_id": request_id,
        "cache": {"version": _CACHE_VERSION, "entries": new_entries},
    }
