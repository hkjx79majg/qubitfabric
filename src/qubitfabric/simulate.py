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

可选的 ``noise`` 描述逐门局部退极化噪声：``x``/``h``/``rx``/``rz``
之后在目标位施加单比特通道，``cx`` 之后在 control/target 两位上施加
双比特通道；子系统 S 上概率 p 的通道为
``D_S(ρ) = (1-p)ρ + p (I_S/2^|S| ⊗ Tr_S ρ)``，顺序与电路 IR 一致。
任一概率非零时改用密度矩阵精确仿真，量子位上限降为 10；``noise``
省略、为 None、为空对象或概率全零时行为与无噪声实现完全一致。
"""

from __future__ import annotations

import math
import random
from typing import Any

from .circuit import bind_parameters, normalize_circuit

__all__ = [
    "SimulationError",
    "estimate_expectation",
    "estimate_grouped_hamiltonian",
    "estimate_gradient",
]

_MAX_QUBITS = 20
_MAX_QUBITS_NOISY = 10
_PAULI_CHARS = frozenset("IXYZ")
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_NOISE_FIELDS = ("single_qubit_depolarizing", "two_qubit_depolarizing")
_NOISE_FIELD_SET = frozenset(_NOISE_FIELDS)


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


def _is_number(value: Any) -> bool:
    """JSON 实数（int 或 float），排除布尔值。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
# 含噪密度矩阵仿真
# ---------------------------------------------------------------------------
#
# 密度矩阵按行主序扁平存储（rho[i * size + j]）。每个门之后按电路顺序
# 施加局部退极化通道 D_S(ρ) = (1-p)ρ + p (I_S/2^|S| ⊗ Tr_S ρ)。

def _dm_apply_single(
    rho: list[complex], size: int, q: int,
    u00: complex, u01: complex, u10: complex, u11: complex,
) -> None:
    """原地执行 ρ ← U ρ U†，U 为作用在量子位 q 上的 2x2 酉矩阵。"""
    step = 1 << q
    cu00 = u00.conjugate()
    cu01 = u01.conjugate()
    cu10 = u10.conjugate()
    cu11 = u11.conjugate()
    for rbase in range(0, size, 2 * step):
        for roff in range(step):
            row0 = (rbase + roff) * size
            row1 = row0 + step * size
            for cbase in range(0, size, 2 * step):
                for coff in range(step):
                    j0 = cbase + coff
                    j1 = j0 + step
                    a = rho[row0 + j0]
                    b = rho[row0 + j1]
                    c = rho[row1 + j0]
                    d = rho[row1 + j1]
                    t00 = u00 * a + u01 * c
                    t01 = u00 * b + u01 * d
                    t10 = u10 * a + u11 * c
                    t11 = u10 * b + u11 * d
                    rho[row0 + j0] = t00 * cu00 + t01 * cu01
                    rho[row0 + j1] = t00 * cu10 + t01 * cu11
                    rho[row1 + j0] = t10 * cu00 + t11 * cu01
                    rho[row1 + j1] = t10 * cu10 + t11 * cu11


def _dm_apply_cx(rho: list[complex], size: int, control: int, target: int) -> None:
    """原地执行 ρ ← C ρ C，C 为 cx 对应的置换（自逆）。"""
    cmask = 1 << control
    tmask = 1 << target
    for i in range(size):
        if (i & cmask) and not (i & tmask):
            ri = i * size
            rj = (i | tmask) * size
            for k in range(size):
                rho[ri + k], rho[rj + k] = rho[rj + k], rho[ri + k]
    for j in range(size):
        if (j & cmask) and not (j & tmask):
            j2 = j | tmask
            for i in range(size):
                base = i * size
                rho[base + j], rho[base + j2] = rho[base + j2], rho[base + j]


def _dm_depolarize_single(rho: list[complex], size: int, q: int, p: float) -> None:
    """原地施加单比特退极化通道：(1-p)ρ + p (I_q/2 ⊗ Tr_q ρ)。"""
    keep = 1.0 - p
    half_p = p / 2.0
    step = 1 << q
    for rbase in range(0, size, 2 * step):
        for roff in range(step):
            row0 = (rbase + roff) * size
            row1 = row0 + step * size
            for cbase in range(0, size, 2 * step):
                for coff in range(step):
                    j0 = cbase + coff
                    j1 = j0 + step
                    a = rho[row0 + j0]
                    b = rho[row0 + j1]
                    c = rho[row1 + j0]
                    d = rho[row1 + j1]
                    mix = (a + d) * half_p
                    rho[row0 + j0] = keep * a + mix
                    rho[row0 + j1] = keep * b
                    rho[row1 + j0] = keep * c
                    rho[row1 + j1] = keep * d + mix


def _dm_depolarize_two(rho: list[complex], size: int, q1: int, q2: int, p: float) -> None:
    """原地施加双比特退极化通道：(1-p)ρ + p (I_{q1,q2}/4 ⊗ Tr_{q1,q2} ρ)。"""
    keep = 1.0 - p
    quarter_p = p / 4.0
    m1 = 1 << q1
    m2 = 1 << q2
    both = m1 | m2
    for i in range(size):
        if i & both:
            continue
        rows = (i * size, (i | m1) * size, (i | m2) * size, (i | both) * size)
        for j in range(size):
            if j & both:
                continue
            cols = (j, j | m1, j | m2, j | both)
            mix = (
                rho[rows[0] + cols[0]]
                + rho[rows[1] + cols[1]]
                + rho[rows[2] + cols[2]]
                + rho[rows[3] + cols[3]]
            ) * quarter_p
            for a in range(4):
                base = rows[a]
                for b in range(4):
                    idx = base + cols[b]
                    rho[idx] = keep * rho[idx] + (mix if a == b else 0j)


def _simulate_noisy(circuit: dict, p1: float, p2: float) -> list[complex]:
    """执行无参数电路并逐门施加退极化通道，返回末态密度矩阵。"""
    size = 1 << circuit["qubit_count"]
    rho = [0j] * (size * size)
    rho[0] = 1 + 0j
    for op in circuit["operations"]:
        gate = op["gate"]
        target = op["target"]
        if gate == "x":
            _dm_apply_single(rho, size, target, 0j, 1 + 0j, 1 + 0j, 0j)
        elif gate == "h":
            _dm_apply_single(rho, size, target, _INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2)
        elif gate == "cx":
            _dm_apply_cx(rho, size, op["control"], target)
            if p2 != 0.0:
                _dm_depolarize_two(rho, size, op["control"], target, p2)
            continue
        elif gate == "rx":
            half = op["angle"] / 2.0
            ms = -1j * math.sin(half)
            _dm_apply_single(rho, size, target, math.cos(half), ms, ms, math.cos(half))
        else:  # rz
            half = op["angle"] / 2.0
            lo = complex(math.cos(half), -math.sin(half))
            hi = complex(math.cos(half), math.sin(half))
            _dm_apply_single(rho, size, target, lo, 0j, 0j, hi)
        if p1 != 0.0:
            _dm_depolarize_single(rho, size, target, p1)
    return rho


def _dm_pauli_expectation(rho: list[complex], size: int, observable: str) -> float:
    """计算 Tr(ρP)，P 由 Pauli 串给出，返回实部并规约到 [-1, 1]。"""
    flip = 0
    for q, ch in enumerate(observable):
        if ch in ("X", "Y"):
            flip |= 1 << q

    total = 0j
    for b in range(size):
        amp = rho[b * size + (b ^ flip)]
        if amp == 0j:
            continue
        phase = 1 + 0j
        for q, ch in enumerate(observable):
            bit = (b >> q) & 1
            if ch == "Y":
                phase *= -1j if bit else 1j
            elif ch == "Z":
                if bit:
                    phase = -phase
        total += amp * phase

    value = total.real
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return _canon_float(value)


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


def _validate_noise(noise: Any) -> tuple[float, float]:
    """校验噪声模型，返回 ``(single_qubit_depolarizing, two_qubit_depolarizing)``。

    ``noise`` 只能是 None 或对象；概率必须是排除 bool 的有限实数且在
    [0, 1] 内，缺省为 0.0。已提供概率按单比特、双比特顺序校验，最后
    拒绝未知字段（多个未知字段按字典序取首个）。
    """
    if noise is None:
        return 0.0, 0.0
    if not isinstance(noise, dict):
        raise SimulationError(
            "invalid_noise_model", "noise",
            "noise must be an object with optional depolarizing probabilities",
        )
    probs: list[float] = []
    for field in _NOISE_FIELDS:
        if field not in noise:
            probs.append(0.0)
            continue
        value = noise[field]
        path = f"noise.{field}"
        if not _is_number(value):
            raise SimulationError(
                "invalid_noise_model", path,
                f"probability at {path} must be a finite number in [0, 1]",
            )
        probability = float(value)
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise SimulationError(
                "invalid_noise_model", path,
                f"probability at {path} must be a finite number in [0, 1]",
            )
        probs.append(probability)
    for key in noise:
        if not isinstance(key, str):
            raise SimulationError(
                "invalid_noise_model", "noise",
                "noise field names must be strings",
            )
    unknown = sorted(key for key in noise if key not in _NOISE_FIELD_SET)
    if unknown:
        name = unknown[0]
        raise SimulationError(
            "invalid_noise_model", f"noise.{name}",
            f"unknown noise field {name!r} at noise.{name}",
        )
    return probs[0], probs[1]


def estimate_expectation(
    circuit: Any,
    observables: Any,
    values: Any = None,
    shots: Any = None,
    seed: Any = None,
    noise: Any = None,
) -> dict:
    """估计电路末态上各 Pauli observable 的期望值，不修改输入。

    电路沿用现有校验与参数绑定语义；``shots`` 省略时返回精确期望值，
    否则每项独立采样并返回 ``{"positive", "negative"}`` 计数。返回
    ``{"qubit_count", "shots", "results"}``，其中 results 按 observables
    输入顺序排列，全部为 JSON 原生类型。

    ``noise`` 可给出逐门局部退极化概率（见模块文档）；任一概率非零时
    按含噪末态的密度矩阵求精确期望值，采样仍按含噪精确值进行。
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

    if noisy:
        rho = _simulate_noisy(bound, p1, p2)
        size = 1 << qubit_count
        exact = [_dm_pauli_expectation(rho, size, obs) for obs in pauli_strings]
    else:
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
# 共享采样预算的 Hamiltonian 联合估计
# ---------------------------------------------------------------------------
#
# terms 沿用优化入口的 observable 与有限 coefficient，按输入顺序贪心分组：
# 每项放入最早的兼容组，两个 Pauli 串仅当每个量子位上的字符相同或至少
# 一方为 I 时兼容。每组只生成一批联合测量样本：测量前把 X/Y 位旋转到
# Z 基，同组各项从相同 bitstring 计算 ±1 本征值。总 shots 在各组间均分
# （余数依组序各加一），各组的采样 RNG 按 ``f"{seed}:group:{index}"``
# 独立派生，互不干扰。


def _compatible(basis: list[str], observable: str) -> bool:
    """observable 是否与组内已有 basis 兼容。

    每个量子位上字符相同或至少一方为 I 才兼容；basis 中尚无测量字符
    （``""`` 占位）的位置视为 I。
    """
    for q, ch in enumerate(observable):
        if ch != "I" and basis[q] != "" and basis[q] != ch:
            return False
    return True


def _group_pauli_strings(
    observables: list[str],
) -> tuple[list[list[str]], list[list[int]], list[list[int]]]:
    """按输入顺序贪心分组，返回 ``(bases, member_indexes, measure_qubits)``。

    每项放入最早的兼容组，无兼容组时新建；组 basis 为各位唯一的非 I
    字符，没有非 I 字符的位置保留 I；``measure_qubits`` 给出各组需要
    旋转到 Z 基并读取的量子位（按编号升序）。
    """
    bases: list[list[str]] = []
    members: list[list[int]] = []
    for index, observable in enumerate(observables):
        for g, basis in enumerate(bases):
            if _compatible(basis, observable):
                for q, ch in enumerate(observable):
                    if ch != "I":
                        basis[q] = ch
                members[g].append(index)
                break
        else:
            basis = [ch if ch != "I" else "" for ch in observable]
            bases.append(basis)
            members.append([index])

    basis_strings = ["".join(ch if ch != "" else "I" for ch in basis) for basis in bases]
    measure_qubits = [[q for q, ch in enumerate(basis) if ch != ""] for basis in bases]
    return basis_strings, members, measure_qubits


def _apply_basis_rotation(state: list[complex], size: int, basis: str) -> None:
    """在状态向量上把 basis 的 X/Y 位旋转到 Z 基，原地修改。

    X 测量前作用 H；Y 测量前作用 rz(-π/2)（即 S†，全局相位无关）再
    作用 H，使随后的 Z 基测量复现原 Pauli 的 ±1 本征值；I/Z 位不动。
    """
    for q, ch in enumerate(basis):
        if ch == "X":
            _apply_h(state, size, q)
        elif ch == "Y":
            _apply_rz(state, size, q, -math.pi / 2.0)
            _apply_h(state, size, q)


def _dm_apply_basis_rotation(rho: list[complex], size: int, basis: str) -> None:
    """密度矩阵版本的基旋转，原地把 X/Y 位转到 Z 基。"""
    for q, ch in enumerate(basis):
        if ch == "X":
            _dm_apply_single(rho, size, q, _INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2)
        elif ch == "Y":
            half = -math.pi / 4.0
            lo = complex(math.cos(half), -math.sin(half))
            hi = complex(math.cos(half), math.sin(half))
            _dm_apply_single(rho, size, q, lo, 0j, 0j, hi)
            _dm_apply_single(rho, size, q, _INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2)


def _diagonal_probabilities(
    amplitudes: Any, size: int, measured: list[int], density_matrix: bool = False,
) -> list[float]:
    """提取 measured 量子位的计算基测量边缘概率。

    ``amplitudes`` 为状态向量（取模平方）或已旋转到测量基的密度矩阵
    （取对角元，由 ``density_matrix`` 区分）；结果按 measured 量子位
    构成的位串索引（measured[0] 为最高位），长度为
    ``2 ** len(measured)``；没有测量位时唯一空串概率恒为 1。
    """
    width = len(measured)
    probabilities = [0.0] * (1 << width)
    if width == 0:
        probabilities[0] = 1.0
        return probabilities
    masks = [1 << q for q in measured]
    for b in range(size):
        outcome = 0
        for k, mask in enumerate(masks):
            if b & mask:
                outcome |= 1 << (width - 1 - k)
        if density_matrix:
            probabilities[outcome] += amplitudes[b * size + b].real
        else:
            amp = amplitudes[b]
            probabilities[outcome] += amp.real * amp.real + amp.imag * amp.imag
    return probabilities


def _sample_group_bitstrings(
    probabilities: list[float], shots: int, seed: int, group_index: int,
) -> list[int]:
    """为一个组抽取一批联合测量样本，返回 measured 位串构成的整数列表。"""
    rng = random.Random(f"{seed}:group:{group_index}")
    outcomes: list[int] = []
    count = len(probabilities)
    for _ in range(shots):
        point = rng.random()
        cumulative = 0.0
        outcome = count - 1
        for k in range(count):
            cumulative += probabilities[k]
            if point < cumulative:
                outcome = k
                break
        outcomes.append(outcome)
    return outcomes


def estimate_grouped_hamiltonian(
    circuit: Any,
    terms: Any,
    values: Any = None,
    shots: Any = None,
    seed: Any = None,
    noise: Any = None,
) -> dict:
    """用一个总采样预算联合估计 Hamiltonian 的各项期望值，不修改输入。

    terms 沿用优化入口的结构（Pauli ``observable`` 加有限实数
    ``coefficient``），按输入顺序贪心分组，兼容的 Pauli 串共享同一批
    联合测量样本。``shots`` 是所有组共享的总预算，按整除结果均分，
    余数依组序各加一；小于分组数时抛 ``insufficient_shots``。

    校验顺序：电路规范化/参数绑定 → terms 结构 → observable 内容 →
    noise → shots → seed；状态空间上限在仿真前检查。返回
    ``{"qubit_count", "shots", "groups", "results", "energy"}``：
    groups 按确定的组序给出 ``{"basis", "shots", "term_indexes"}``；
    results 与 terms 一一对应（保留重复项），每项为
    ``{"observable", "coefficient", "expectation", "plus_count",
    "minus_count"}``；energy 为 coefficient×expectation 按输入顺序之和。
    """
    normalized = normalize_circuit(circuit)
    bound = bind_parameters(normalized, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    # terms 结构校验沿用优化入口（抛 OptimizationError）；延迟导入避开
    # optimize → simulate 的模块级循环依赖。
    from .optimize import _validate_terms
    observables, coefficients = _validate_terms(terms)
    _validate_observables(observables, qubit_count)
    p1, p2 = _validate_noise(noise)
    bases, members, measured = _group_pauli_strings(observables)
    group_count = len(bases)
    shot_count = _validate_total_shots(shots, group_count)
    resolved_seed = _validate_grouped_seed(seed)

    noisy = p1 != 0.0 or p2 != 0.0
    max_qubits = _MAX_QUBITS_NOISY if noisy else _MAX_QUBITS
    if qubit_count > max_qubits:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {max_qubits}-qubit simulation limit",
        )

    size = 1 << qubit_count
    base, remainder = divmod(shot_count, group_count)
    group_shots = [base + (1 if g < remainder else 0) for g in range(group_count)]
    plus_counts = [0] * len(observables)
    minus_counts = [0] * len(observables)

    for g in range(group_count):
        basis = bases[g]
        if noisy:
            amplitudes: Any = _simulate_noisy(bound, p1, p2)
            _dm_apply_basis_rotation(amplitudes, size, basis)
            probabilities = _diagonal_probabilities(amplitudes, size, measured[g], True)
        else:
            amplitudes = _simulate(bound)
            _apply_basis_rotation(amplitudes, size, basis)
            probabilities = _diagonal_probabilities(amplitudes, size, measured[g])
        samples = _sample_group_bitstrings(probabilities, group_shots[g], resolved_seed, g)

        width = len(measured[g])
        bit_position = {q: width - 1 - k for k, q in enumerate(measured[g])}
        for term_index in members[g]:
            observable = observables[term_index]
            if not any(ch != "I" for ch in observable):
                plus_counts[term_index] = group_shots[g]
                continue
            plus = 0
            for sample in samples:
                parity = 0
                for q, ch in enumerate(observable):
                    if ch != "I" and ((sample >> bit_position[q]) & 1):
                        parity ^= 1
                if parity == 0:
                    plus += 1
            plus_counts[term_index] = plus
            minus_counts[term_index] = group_shots[g] - plus

    results: list[dict] = []
    energy = 0.0
    for k, observable in enumerate(observables):
        total = plus_counts[k] + minus_counts[k]
        expectation = _canon_float((plus_counts[k] - minus_counts[k]) / total)
        energy += coefficients[k] * expectation
        results.append({
            "observable": observable,
            "coefficient": coefficients[k],
            "expectation": expectation,
            "plus_count": plus_counts[k],
            "minus_count": minus_counts[k],
        })

    groups = [
        {"basis": bases[g], "shots": group_shots[g], "term_indexes": members[g]}
        for g in range(group_count)
    ]
    return {
        "qubit_count": qubit_count,
        "shots": shot_count,
        "groups": groups,
        "results": results,
        "energy": _canon_float(energy),
    }


def _validate_total_shots(shots: Any, group_count: int) -> int:
    """校验总采样预算：必须是排除 bool 的正整数且不小于分组数。"""
    if not _is_int(shots) or shots <= 0:
        raise SimulationError("invalid_shots", "shots", "shots must be a positive integer")
    if shots < group_count:
        raise SimulationError(
            "insufficient_shots", "shots",
            f"shots {shots} cannot cover {group_count} measurement groups",
        )
    return shots


def _validate_grouped_seed(seed: Any) -> int:
    """seed 省略为 0；显式值必须是排除 bool 的非负整数。"""
    if seed is None:
        return 0
    if not _is_int(seed) or seed < 0:
        raise SimulationError("invalid_seed", "seed", "seed must be a non-negative integer")
    return seed


# ---------------------------------------------------------------------------
# 参数移位梯度
# ---------------------------------------------------------------------------

_HALF_PI = math.pi / 2.0


def _simulate_with_shift(
    bound: dict,
    op_index: int,
    shift: float,
    p1: float = 0.0,
    p2: float = 0.0,
) -> list[complex]:
    """在绑定电路基础上把第 op_index 个旋转门角度平移 shift 后仿真。"""
    operations = [dict(op) for op in bound["operations"]]
    operations[op_index]["angle"] = operations[op_index]["angle"] + shift
    circuit = {
        "qubit_count": bound["qubit_count"],
        "parameters": [],
        "operations": operations,
    }
    if p1 != 0.0 or p2 != 0.0:
        return _simulate_noisy(circuit, p1, p2)
    return _simulate(circuit)


def estimate_gradient(circuit: Any, observables: Any, values: Any = None, noise: Any = None) -> dict:
    """精确参数移位梯度：各 observable 对全部声明参数的导数，不修改输入。

    校验与失败语义和精确 :func:`estimate_expectation` 一致。角度为
    ``aθ+b`` 的旋转门对参数 θ 的单次贡献为 ``a/2`` 乘以该门角度单独
    增减 ``π/2`` 后两次精确期望值之差；同一参数出现在多个旋转门时
    累加各门贡献，未用于参数化旋转的参数导数为 ``0.0``。``noise``
    非零时期望值与导数都按含噪密度矩阵精确计算（噪声通道与参数无关，
    参数移位规则保持不变）。

    返回 ``{"qubit_count", "parameters", "results"}``：parameters 保持
    声明顺序；results 保持 observables 输入顺序（含重复项），每项为
    ``{"observable", "expectation", "gradients"}``，gradients 按
    parameters 顺序给出浮点导数；所有输出把 ``-0.0`` 规范为 ``0.0``。
    """
    normalized = normalize_circuit(circuit)
    bound = bind_parameters(normalized, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    pauli_strings = _validate_observables(observables, qubit_count)
    p1, p2 = _validate_noise(noise)

    noisy = p1 != 0.0 or p2 != 0.0
    max_qubits = _MAX_QUBITS_NOISY if noisy else _MAX_QUBITS
    if qubit_count > max_qubits:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {max_qubits}-qubit simulation limit",
        )

    parameters = list(normalized["parameters"])
    size = 1 << qubit_count

    if noisy:
        def _expect(sim_result: list[complex], obs: str) -> float:
            return _dm_pauli_expectation(sim_result, size, obs)

        base_state = _simulate_noisy(bound, p1, p2)
    else:
        def _expect(sim_result: list[complex], obs: str) -> float:
            return _pauli_expectation(sim_result, obs)

        base_state = _simulate(bound)
    base = [_expect(base_state, obs) for obs in pauli_strings]

    # 绑定前后操作一一对应；收集各参数出现的旋转门（下标与线性系数）。
    contributions: dict[str, list[tuple[int, float]]] = {name: [] for name in parameters}
    for i, op in enumerate(normalized["operations"]):
        angle = op.get("angle")
        if isinstance(angle, dict):
            contributions[angle["parameter"]].append((i, angle["coefficient"]))

    totals = {name: [0.0] * len(pauli_strings) for name in parameters}
    for name in parameters:
        for op_index, coefficient in contributions[name]:
            plus_state = _simulate_with_shift(bound, op_index, _HALF_PI, p1, p2)
            minus_state = _simulate_with_shift(bound, op_index, -_HALF_PI, p1, p2)
            factor = coefficient / 2.0
            for k, obs in enumerate(pauli_strings):
                plus = _expect(plus_state, obs)
                minus = _expect(minus_state, obs)
                totals[name][k] += factor * (plus - minus)

    results: list[dict] = []
    for k, obs in enumerate(pauli_strings):
        gradients = {name: _canon_float(totals[name][k]) for name in parameters}
        results.append({"observable": obs, "expectation": base[k], "gradients": gradients})

    return {"qubit_count": qubit_count, "parameters": parameters, "results": results}
