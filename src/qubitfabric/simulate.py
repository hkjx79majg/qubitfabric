"""无噪声状态向量仿真与 Pauli 期望值估计。

仿真从全零态出发，qubit 0 对应状态索引的最低有效位；旋转门采用
``exp(-iθP/2)`` 约定（``rx(π)`` 把 ``|0⟩`` 映到 ``-i|1⟩``）。
observable 是长度等于 ``qubit_count`` 的 Pauli 串，字符 ``I/X/Y/Z``
的下标即量子位编号，结果按输入顺序返回。

省略 ``shots`` 时返回精确浮点期望值；给出 ``shots`` 时对每个
observable 独立采样 ``shots`` 次，返回正一/负一计数及由计数得到的
期望值。相同输入与 ``seed`` 产生完全相同的结果；采样 RNG 按
``f"{seed}:{index}"`` 派生，各项计数只依赖 seed、序号与 shots。

:func:`estimate_gradient` 在精确期望值基础上给出参数移位梯度：
仅覆盖 rx/rz 的线性参数角，按声明参数顺序返回各 observable 的
精确导数。
"""

from __future__ import annotations

import math
import random
from typing import Any

from .circuit import bind_parameters, normalize_circuit

__all__ = ["SimulationError", "estimate_expectation", "estimate_gradient"]

_MAX_QUBITS = 20
_PAULI_CHARS = frozenset("IXYZ")
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class SimulationError(ValueError):
    """仿真入口校验失败。

    ``code`` 为稳定的机器可读错误码，``path`` 指向输入中出错的位置，
    语义与 :class:`qubitfabric.circuit.CircuitValidationError` 一致。
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


def _canon_float(value: float) -> float:
    """统一浮点形态：抹掉 -0.0。"""
    if value == 0.0:
        return 0.0
    return value


# ---------------------------------------------------------------------------
# 状态向量仿真
# ---------------------------------------------------------------------------

def _apply_x(state: list[complex], size: int, q: int) -> None:
    step = 1 << q
    for base in range(0, size, 2 * step):
        for off in range(step):
            i = base + off
            j = i + step
            state[i], state[j] = state[j], state[i]


def _apply_h(state: list[complex], size: int, q: int) -> None:
    step = 1 << q
    for base in range(0, size, 2 * step):
        for off in range(step):
            i = base + off
            j = i + step
            a = state[i]
            b = state[j]
            state[i] = (a + b) * _INV_SQRT2
            state[j] = (a - b) * _INV_SQRT2


def _apply_rx(state: list[complex], size: int, q: int, angle: float) -> None:
    half = angle / 2.0
    c = math.cos(half)
    ms = -1j * math.sin(half)
    step = 1 << q
    for base in range(0, size, 2 * step):
        for off in range(step):
            i = base + off
            j = i + step
            a = state[i]
            b = state[j]
            state[i] = c * a + ms * b
            state[j] = ms * a + c * b


def _apply_rz(state: list[complex], size: int, q: int, angle: float) -> None:
    half = angle / 2.0
    lo = complex(math.cos(half), -math.sin(half))
    hi = complex(math.cos(half), math.sin(half))
    step = 1 << q
    for base in range(0, size, 2 * step):
        for off in range(step):
            i = base + off
            state[i] *= lo
            state[i + step] *= hi


def _apply_cx(state: list[complex], size: int, control: int, target: int) -> None:
    cmask = 1 << control
    tmask = 1 << target
    for i in range(size):
        if (i & cmask) and not (i & tmask):
            j = i | tmask
            state[i], state[j] = state[j], state[i]


def _simulate(circuit: dict) -> list[complex]:
    """执行无参数电路，返回末态状态向量。"""
    size = 1 << circuit["qubit_count"]
    state = [0j] * size
    state[0] = 1 + 0j
    for op in circuit["operations"]:
        gate = op["gate"]
        if gate == "x":
            _apply_x(state, size, op["target"])
        elif gate == "h":
            _apply_h(state, size, op["target"])
        elif gate == "cx":
            _apply_cx(state, size, op["control"], op["target"])
        elif gate == "rx":
            _apply_rx(state, size, op["target"], op["angle"])
        else:  # rz
            _apply_rz(state, size, op["target"], op["angle"])
    return state


# ---------------------------------------------------------------------------
# Pauli 期望值
# ---------------------------------------------------------------------------

def _pauli_expectation(state: list[complex], observable: str) -> float:
    """计算 ``⟨ψ|P|ψ⟩``，P 由 Pauli 串给出，返回实部并规约到 [-1, 1]。"""
    flip = 0
    for q, ch in enumerate(observable):
        if ch in ("X", "Y"):
            flip |= 1 << q

    total = 0j
    for b, amp in enumerate(state):
        if amp == 0j:
            continue
        c = b ^ flip
        phase = 1 + 0j
        for q, ch in enumerate(observable):
            bit = (c >> q) & 1
            if ch == "Y":
                phase *= -1j if bit else 1j
            elif ch == "Z":
                if bit:
                    phase = -phase
        total += amp.conjugate() * phase * state[c]

    value = total.real
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return _canon_float(value)


# ---------------------------------------------------------------------------
# 入口校验与估计
# ---------------------------------------------------------------------------

def _validate_observables(observables: Any, qubit_count: int) -> list[str]:
    if not isinstance(observables, list) or not observables:
        raise SimulationError(
            "invalid_observables", "observables",
            "observables must be a non-empty array of Pauli strings",
        )
    for i, obs in enumerate(observables):
        path = f"observables[{i}]"
        if not isinstance(obs, str):
            raise SimulationError("invalid_observable", path, f"observable at {path} must be a string")
        if len(obs) != qubit_count:
            raise SimulationError(
                "invalid_observable", path,
                f"observable at {path} must have length {qubit_count}",
            )
        if any(ch not in _PAULI_CHARS for ch in obs):
            raise SimulationError(
                "invalid_observable", path,
                f"observable at {path} may only contain I, X, Y, Z",
            )
    return list(observables)


def _validate_shots(shots: Any) -> int | None:
    if shots is None:
        return None
    if not _is_int(shots) or shots <= 0:
        raise SimulationError("invalid_shots", "shots", "shots must be a positive integer")
    return shots


def _validate_seed(seed: Any, shots: int | None) -> int | None:
    if shots is None:
        if seed is not None:
            raise SimulationError(
                "seed_without_shots", "seed",
                "seed is only meaningful together with shots",
            )
        return None
    if seed is None:
        return 0
    if not _is_int(seed) or seed < 0:
        raise SimulationError("invalid_seed", "seed", "seed must be a non-negative integer")
    return seed


def estimate_expectation(
    circuit: Any,
    observables: Any,
    values: Any = None,
    shots: Any = None,
    seed: Any = None,
) -> dict:
    """估计电路末态上各 Pauli observable 的期望值，不修改输入。

    电路沿用现有校验与参数绑定语义；``shots`` 省略时返回精确期望值，
    否则每项独立采样并返回 ``{"positive", "negative"}`` 计数。返回
    ``{"qubit_count", "shots", "results"}``，其中 results 按 observables
    输入顺序排列，全部为 JSON 原生类型。
    """
    bound = bind_parameters(circuit, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    pauli_strings = _validate_observables(observables, qubit_count)
    shot_count = _validate_shots(shots)
    resolved_seed = _validate_seed(seed, shot_count)

    if qubit_count > _MAX_QUBITS:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {_MAX_QUBITS}-qubit state vector limit",
        )

    state = _simulate(bound)
    exact = [_pauli_expectation(state, obs) for obs in pauli_strings]

    results: list[dict] = []
    if shot_count is None:
        for obs, value in zip(pauli_strings, exact):
            results.append({"observable": obs, "expectation": value})
    else:
        for i, (obs, value) in enumerate(zip(pauli_strings, exact)):
            rng = random.Random(f"{resolved_seed}:{i}")
            p_plus = (1.0 + value) / 2.0
            positive = sum(1 for _ in range(shot_count) if rng.random() < p_plus)
            negative = shot_count - positive
            results.append({
                "observable": obs,
                "expectation": _canon_float((positive - negative) / shot_count),
                "counts": {"positive": positive, "negative": negative},
            })

    return {"qubit_count": qubit_count, "shots": shot_count, "results": results}


# ---------------------------------------------------------------------------
# 参数移位梯度
# ---------------------------------------------------------------------------

_HALF_PI = math.pi / 2.0


def _simulate_with_shift(bound: dict, op_index: int, shift: float) -> list[complex]:
    """在绑定电路基础上把第 op_index 个旋转门角度平移 shift 后仿真。"""
    operations = [dict(op) for op in bound["operations"]]
    operations[op_index]["angle"] = operations[op_index]["angle"] + shift
    return _simulate({
        "qubit_count": bound["qubit_count"],
        "parameters": [],
        "operations": operations,
    })


def estimate_gradient(circuit: Any, observables: Any, values: Any = None) -> dict:
    """精确参数移位梯度：各 observable 对全部声明参数的导数，不修改输入。

    校验与失败语义和精确 :func:`estimate_expectation` 一致。角度为
    ``aθ+b`` 的旋转门对参数 θ 的单次贡献为 ``a/2`` 乘以该门角度单独
    增减 ``π/2`` 后两次精确期望值之差；同一参数出现在多个旋转门时
    累加各门贡献，未用于参数化旋转的参数导数为 ``0.0``。

    返回 ``{"qubit_count", "parameters", "results"}``：parameters 保持
    声明顺序；results 保持 observables 输入顺序（含重复项），每项为
    ``{"observable", "expectation", "gradients"}``，gradients 按
    parameters 顺序给出浮点导数；所有输出把 ``-0.0`` 规范为 ``0.0``。
    """
    normalized = normalize_circuit(circuit)
    bound = bind_parameters(normalized, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    pauli_strings = _validate_observables(observables, qubit_count)

    if qubit_count > _MAX_QUBITS:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {_MAX_QUBITS}-qubit state vector limit",
        )

    parameters = list(normalized["parameters"])

    base_state = _simulate(bound)
    base = [_pauli_expectation(base_state, obs) for obs in pauli_strings]

    # 绑定前后操作一一对应；收集各参数出现的旋转门（下标与线性系数）。
    contributions: dict[str, list[tuple[int, float]]] = {name: [] for name in parameters}
    for i, op in enumerate(normalized["operations"]):
        angle = op.get("angle")
        if isinstance(angle, dict):
            contributions[angle["parameter"]].append((i, angle["coefficient"]))

    totals = {name: [0.0] * len(pauli_strings) for name in parameters}
    for name in parameters:
        for op_index, coefficient in contributions[name]:
            plus_state = _simulate_with_shift(bound, op_index, _HALF_PI)
            minus_state = _simulate_with_shift(bound, op_index, -_HALF_PI)
            factor = coefficient / 2.0
            for k, obs in enumerate(pauli_strings):
                plus = _pauli_expectation(plus_state, obs)
                minus = _pauli_expectation(minus_state, obs)
                totals[name][k] += factor * (plus - minus)

    results: list[dict] = []
    for k, obs in enumerate(pauli_strings):
        gradients = {name: _canon_float(totals[name][k]) for name in parameters}
        results.append({"observable": obs, "expectation": base[k], "gradients": gradients})

    return {"qubit_count": qubit_count, "parameters": parameters, "results": results}
