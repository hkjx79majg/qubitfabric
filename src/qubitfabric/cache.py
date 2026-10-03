"""跨进程稳定的期望值结果缓存。

:func:`cached_expectation` 接收与
:func:`qubitfabric.simulate.estimate_expectation` 相同的基础输入
（circuit、observables、values、shots、seed、noise），另接收可选的
``cache`` 快照与默认 128 的 ``max_entries``，返回
``{"result", "cache_hit", "request_id", "cache"}``：

- ``result`` 与相同输入直接调用 :meth:`Service.expectation` 完全一致；
- ``request_id`` 是请求身份的 64 位小写十六进制 SHA-256；
- ``cache`` 为全新对象，仅含 JSON 原生类型，``json.dumps``/
  ``json.loads`` 往返后可跨进程传回继续命中，全程不修改任何输入。

请求身份由完整绑定后的规范电路、保留顺序与重复项的 observables、
解析后的 shots、seed 及补齐默认值的 noise 组成，按键排序、紧凑分隔、
UTF-8 编码的 JSON 计算。省略默认值或使用等价数字形式不改变身份；
observable 顺序、绑定结果、采样数、种子或噪声变化都会未命中。

缓存固定形如 ``{"version": 1, "entries": [...]}``，条目仅含
``request_id``、``result``、``result_digest``，按最近使用优先排列，
``result_digest`` 为 result 规范 JSON 的 SHA-256。命中时核对摘要、
返回独立副本并把条目移到首位；未命中时按现有语义计算并插入首部，
超出 ``max_entries`` 时从末项淘汰。采样未给 seed 时继续沿用现有
默认种子。

校验顺序：基础请求沿用 ``expectation`` 的校验顺序、异常类型与量子位
限制，其后校验 ``max_entries`` 与 ``cache``。``max_entries`` 不是
排除 bool 的正整数时抛 :class:`CacheStateError`
（``invalid_cache_capacity`` / ``max_entries``）；cache 非对象、版本
错误、字段缺失或未知、entries 或条目结构错误、request_id 重复或格式
错误、摘要格式错误或与 result 不符时抛 :class:`CacheStateError`
（``invalid_cache`` / ``cache``）。结构合法但不含当前身份只是普通
未命中。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

from .simulate import _compute_expectation, _prepare_expectation

__all__ = ["CacheStateError", "cached_expectation"]

_CACHE_VERSION = 1
_DEFAULT_MAX_ENTRIES = 128
_CACHE_FIELDS = ("version", "entries")
_CACHE_FIELD_SET = frozenset(_CACHE_FIELDS)
_ENTRY_FIELDS = ("request_id", "result", "result_digest")
_ENTRY_FIELD_SET = frozenset(_ENTRY_FIELDS)
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class CacheStateError(ValueError):
    """期望值缓存的容量或快照状态校验失败。

    ``code`` 为稳定的机器可读错误码（``invalid_cache_capacity``、
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


def _invalid(message: str) -> CacheStateError:
    return CacheStateError("invalid_cache", "cache", message)


def _is_int(value: Any) -> bool:
    """JSON 整数；bool 是 int 的子类，但不算数值。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_max_entries(max_entries: Any) -> int:
    """必须是排除 bool 的正整数，否则报 invalid_cache_capacity。"""
    if not _is_int(max_entries) or max_entries < 1:
        raise CacheStateError(
            "invalid_cache_capacity", "max_entries",
            "max_entries must be a positive integer",
        )
    return max_entries


# ---------------------------------------------------------------------------
# 请求身份与结果摘要
# ---------------------------------------------------------------------------

def _canonical_digest(value: Any) -> str:
    """按键排序、紧凑分隔、UTF-8 的规范 JSON 的 SHA-256 小写十六进制。"""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request_id_of(prepared: dict) -> str:
    """由规范化后的完整请求计算跨进程稳定身份。"""
    payload = {
        "circuit": prepared["bound"],
        "observables": prepared["observables"],
        "shots": prepared["shots"],
        "seed": prepared["seed"],
        "noise": prepared["noise"],
    }
    return _canonical_digest(payload)


# ---------------------------------------------------------------------------
# 快照校验
# ---------------------------------------------------------------------------

def _is_json_native(value: Any) -> bool:
    """递归判断是否仅含 JSON 原生类型（数字必须有限，键必须为字符串）。"""
    if value is None or isinstance(value, (str, bool, int)):
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
    """校验缓存快照并返回条目构成的全新列表（深拷贝，不共享输入对象）。

    任何结构错误都抛 ``invalid_cache`` / ``cache``；结构合法但摘要
    与 request_id 互不匹配的快照同样被拒绝。
    """
    if cache is None:
        return []
    if not isinstance(cache, dict):
        raise _invalid("cache must be an object")

    # 先按输入（字典插入）顺序拒绝未知字段。
    for key in cache:
        if not isinstance(key, str) or key not in _CACHE_FIELD_SET:
            raise _invalid(f"unknown cache field {key!r}")
    for field in _CACHE_FIELDS:
        if field not in cache:
            raise _invalid(f"missing cache field {field!r}")

    version = cache["version"]
    if not _is_int(version) or version != _CACHE_VERSION:
        raise _invalid(f"unsupported cache version {version!r}")

    raw_entries = cache["entries"]
    if not isinstance(raw_entries, list):
        raise _invalid("cache entries must be an array")

    entries: list[dict] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        location = f"cache.entries[{index}]"
        if not isinstance(raw_entry, dict):
            raise _invalid(f"entry at {location} must be an object")
        for key in raw_entry:
            if not isinstance(key, str) or key not in _ENTRY_FIELD_SET:
                raise _invalid(f"unknown entry field {key!r} at {location}")
        for field in _ENTRY_FIELDS:
            if field not in raw_entry:
                raise _invalid(f"missing entry field {field!r} at {location}")

        request_id = raw_entry["request_id"]
        if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
            raise _invalid(f"request_id at {location} must be 64 lowercase hex characters")
        if request_id in seen:
            raise _invalid(f"duplicate request_id {request_id!r} at {location}")

        result = raw_entry["result"]
        if not _is_json_native(result):
            raise _invalid(f"result at {location} must contain only JSON-native values")

        digest = raw_entry["result_digest"]
        if not isinstance(digest, str) or not _REQUEST_ID_RE.fullmatch(digest):
            raise _invalid(f"result_digest at {location} must be 64 lowercase hex characters")
        if digest != _canonical_digest(result):
            raise _invalid(f"result_digest at {location} does not match result")

        seen.add(request_id)
        entries.append(copy.deepcopy(raw_entry))

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
    max_entries: Any = _DEFAULT_MAX_ENTRIES,
) -> dict:
    """带跨进程稳定缓存的期望值估计，不修改输入。

    返回 ``{"result", "cache_hit", "request_id", "cache"}``：基础请求
    的校验、异常与量子位限制与 :func:`estimate_expectation` 相同；
    命中时 ``cache_hit`` 为 True、``result`` 为存储结果的独立副本，
    未命中时按现有语义计算并把新条目插到首位、按 ``max_entries``
    从末项淘汰。返回的 cache 与 result 都是不与任何输入或彼此共享的
    新对象。
    """
    prepared = _prepare_expectation(
        circuit, observables, values=values, shots=shots, seed=seed, noise=noise,
    )
    capacity = _validate_max_entries(max_entries)
    entries = _validate_cache(cache)
    request_id = _request_id_of(prepared)

    for index, entry in enumerate(entries):
        if entry["request_id"] != request_id:
            continue
        # 载入时已核对过摘要；命中时再次核对，拒绝任何不一致状态。
        if entry["result_digest"] != _canonical_digest(entry["result"]):
            raise _invalid(f"result_digest at cache.entries[{index}] does not match result")
        result = copy.deepcopy(entry["result"])
        entries.insert(0, entries.pop(index))
        new_cache = {"version": _CACHE_VERSION, "entries": entries}
        return {
            "result": result,
            "cache_hit": True,
            "request_id": request_id,
            "cache": new_cache,
        }

    result = _compute_expectation(prepared)
    entries.insert(0, {
        "request_id": request_id,
        "result": copy.deepcopy(result),
        "result_digest": _canonical_digest(result),
    })
    if len(entries) > capacity:
        del entries[capacity:]
    new_cache = {"version": _CACHE_VERSION, "entries": entries}
    return {
        "result": copy.deepcopy(result),
        "cache_hit": False,
        "request_id": request_id,
        "cache": new_cache,
    }
