"""无噪声状态向量仿真与 Pauli 期望值估计。

仿真从全零态出发，qubit 0 对应状态索引的最低有效位；旋转门采用
``exp(-iθP/2)`` 约定。观测量为 Pauli 串（``I``/``X``/``Y``/``Z``），
字符串下标即量子位编号，长度必须等于 ``qubit_count``。

省略 ``shots`` 时返回精确期望值；给出 ``shots`` 时对每个观测量独立采样，
返回 ±1 计数与由计数得到的期望值。相同输入与 ``seed`` 产生完全相同的结果。
"""

from __future__ import annotations

import math
import random
from typing import Any

from .circuit import (
    _canon_float,
    _is_int,
    bind_parameters,
    normalize_circuit,
)

__all__ = ["SimulationError", "estimate_expectation"]

_MAX_QUBITS = 20
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_PAULIS = frozenset("IXYZ")


class SimulationError(ValueError):
    """仿真请求校验失败。

    ``code`` 为稳定的机器可读错误码，``path`` 指向请求中出错的位置
    （如 ``"observables[0]"``、``"shots"``）。
    """

    def __init__(self, code: str, path: str, message: str | None = None) -> None:
        self.code = code
        self.path = path
        if message is None:
            message = f"{code} at {path}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# 请求校验
# ---------------------------------------------------------------------------

def _validate_observables(observables: Any, qubit_count: int) -> list[str]:
    if not isinstance(observables, list) or not observables:
        raise SimulationError(
            "invalid_observables", "observables", "observables must be a non-empty array of Pauli strings"
        )
    result: list[str] = []
    for i, item in enumerate(observables):
        if (
            not isinstance(item, str)
            or len(item) != qubit_count
            or any(ch not in _PAULIS for ch in item)
        ):
            raise SimulationError(
                "invalid_observable",
                f"observables[{i}]",
                f"observable at observables[{i}] must be a Pauli string of length {qubit_count}",
            )
        result.append(item)
    return result


def _validate_shots(shots: Any) -> None:
    if shots is None:
        return
    if not _is_int(shots) or shots <= 0:
        raise SimulationError("invalid_shots", "shots", "shots must be a positive integer")


def _validate_seed(shots: Any, seed: Any) -> None:
    if shots is None:
        if seed is not None:
            raise SimulationError("seed_without_shots", "seed", "seed requires shots to be set")
        return
    if seed is not None and (not _is_int(seed) or seed < 0):
        raise SimulationError("invalid_seed", "seed", "seed must be a non-negative integer")


# ---------------------------------------------------------------------------
# 状态向量仿真
# ---------------------------------------------------------------------------

def _apply_x(state: list[complex], mask: int) -> None:
    step = mask << 1
    for base in range(0, len(state), step):
        for i in range(base, base + mask):
            j = i + mask
            state[i], state[j] = state[j], state[i]


def _apply_h(state: list[complex], mask: int) -> None:
    step = mask << 1
    for base in range(0, len(state), step):
        for i in range(base, base + mask):
            j = i + mask
            a = state[i]
            b = state[j]
            state[i] = (a + b) * _INV_SQRT2
            state[j] = (a - b) * _INV_SQRT2


def _apply_rx(state: list[complex], mask: int, angle: float) -> None:
    half = angle / 2.0
    c = math.cos(half)
    ms = complex(0.0, -math.sin(half))
    step = mask << 1
    for base in range(0, len(state), step):
        for i in range(base, base + mask):
            j = i + mask
            a = state[i]
            b = state[j]
            state[i] = c * a + ms * b
            state[j] = ms * a + c * b


def _apply_rz(state: list[complex], mask: int, angle: float) -> None:
    half = angle / 2.0
    lo = complex(math.cos(half), -math.sin(half))
    hi = complex(math.cos(half), math.sin(half))
    step = mask << 1
    for base in range(0, len(state), step):
        for i in range(base, base + mask):
            state[i] *= lo
            state[i + mask] *= hi


def _apply_cx(state: list[complex], control_mask: int, target_mask: int) -> None:
    for i in range(len(state)):
        if i & control_mask and not i & target_mask:
            j = i + target_mask
            state[i], state[j] = state[j], state[i]


def _simulate(operations: list[dict], qubit_count: int) -> list[complex]:
    state = [0j] * (1 << qubit_count)
    state[0] = 1.0 + 0j
    for op in operations:
        gate = op["gate"]
        if gate == "x":
            _apply_x(state, 1 << op["target"])
        elif gate == "h":
            _apply_h(state, 1 << op["target"])
        elif gate == "rx":
            _apply_rx(state, 1 << op["target"], op["angle"])
        elif gate == "rz":
            _apply_rz(state, 1 << op["target"], op["angle"])
        else:  # cx
            _apply_cx(state, 1 << op["control"], 1 << op["target"])
    return state


def _pauli_expectation(state: list[complex], observable: str) -> float:
    """计算 ``⟨ψ|P|ψ⟩``，结果规约到 [-1, 1] 且零统一为 0.0。"""
    size = len(state)
    flip = 0
    y_count = 0
    # 每个量子位「位为 1 相对位为 0」的相位比（X→1，Y→-1，Z→-1，I→1）。
    factors = [1] * len(observable)
    for q, pauli in enumerate(observable):
        if pauli == "X":
            flip |= 1 << q
        elif pauli == "Y":
            flip |= 1 << q
            y_count += 1
            factors[q] = -1
        elif pauli == "Z":
            factors[q] = -1

    # P|i⟩ = phase[i] · |i ^ flip⟩；Y 在位为 0 时贡献 +i（基准相位 i^y_count），
    # 位为 1 时贡献 -i（比值 -1），phase 按最低置位递推构造。
    phase: list[Any] = [1j ** y_count] + [1] * (size - 1)
    for i in range(1, size):
        lsb = i & (-i)
        phase[i] = phase[i ^ lsb] * factors[lsb.bit_length() - 1]

    acc = 0j
    for i in range(size):
        acc += state[i ^ flip].conjugate() * state[i] * phase[i]

    value = min(1.0, max(-1.0, acc.real))
    return _canon_float(value)


# ---------------------------------------------------------------------------
# 期望值估计入口
# ---------------------------------------------------------------------------

def estimate_expectation(
    circuit: Any,
    observables: Any,
    values: Any = None,
    shots: Any = None,
    seed: Any = None,
) -> dict:
    """对电路做无噪声状态向量仿真，按输入顺序返回各 Pauli 观测量的期望值。

    电路校验与参数绑定复用 :mod:`qubitfabric.circuit`；``shots`` 省略时
    返回精确浮点期望值，否则对每个观测量独立采样 ``shots`` 次，返回 ±1
    计数与由计数得到的期望值。输入对象均不被修改。
    """
    bound = bind_parameters(normalize_circuit(circuit), {} if values is None else values)
    paulis = _validate_observables(observables, bound["qubit_count"])
    _validate_shots(shots)
    _validate_seed(shots, seed)

    qubit_count = bound["qubit_count"]
    if qubit_count > _MAX_QUBITS:
        raise SimulationError(
            "state_space_too_large", "qubit_count", f"qubit_count {qubit_count} exceeds {_MAX_QUBITS}"
        )

    state = _simulate(bound["operations"], qubit_count)
    exact = [_pauli_expectation(state, pauli) for pauli in paulis]

    if shots is None:
        return {
            "mode": "exact",
            "results": [
                {"observable": pauli, "expectation": value} for pauli, value in zip(paulis, exact)
            ],
        }

    rng = random.Random(seed)
    results = []
    for pauli, value in zip(paulis, exact):
        p_plus = min(1.0, max(0.0, (1.0 + value) / 2.0))
        plus_one = sum(1 for _ in range(shots) if rng.random() < p_plus)
        minus_one = shots - plus_one
        results.append(
            {
                "observable": pauli,
                "expectation": _canon_float((plus_one - minus_one) / shots),
                "counts": {"plus_one": plus_one, "minus_one": minus_one},
            }
        )
    return {"mode": "sampled", "shots": shots, "results": results}
