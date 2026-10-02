"""状态向量/密度矩阵仿真与 Pauli 期望值估计。

仿真从全零态出发，qubit 0 对应状态索引的最低有效位；旋转门采用
``exp(-iθP/2)`` 约定（``rx(π)`` 把 ``|0⟩`` 映到 ``-i|1⟩``）。
observable 是长度等于 ``qubit_count`` 的 Pauli 串，字符 ``I/X/Y/Z``
的下标即量子位编号，结果按输入顺序返回。

省略 ``shots`` 时返回精确浮点期望值；给出 ``shots`` 时对每个
observable 独立采样 ``shots`` 次，返回正一/负一计数及由计数得到的
期望值。相同输入与 ``seed`` 产生完全相同的结果；采样 RNG 按
``f"{seed}:{index}"`` 派生，各项计数只依赖 seed、序号与 shots。

可选的 ``noise`` 描述逐门局部退极化噪声：``None``、缺省或概率全零
时与无噪声行为完全一致；否则在 x/h/rx/rz 后对 target 施加单比特
通道、在 cx 后对 control 与 target 子系统施加双比特通道，子系统 S
上概率 p 的通道为 ``D_S(ρ) = (1-p)ρ + p (I_S/2^|S| ⊗ Tr_S ρ)``，
施加顺序与电路 IR 一致。含噪仿真改用密度矩阵，qubit 上限从 20
降为 10。

:func:`estimate_gradient` 在精确期望值基础上给出参数移位梯度：
仅覆盖 rx/rz 的线性参数角，按声明参数顺序返回各 observable 的
精确导数；含噪时基准值与移位期望值都来自含噪密度矩阵。
"""

from __future__ import annotations

import math
import random
from typing import Any

from .circuit import bind_parameters, normalize_circuit

__all__ = ["SimulationError", "estimate_expectation", "estimate_gradient"]

_MAX_QUBITS = 20
_MAX_QUBITS_NOISY = 10
_PAULI_CHARS = frozenset("IXYZ")
_INV_SQRT2 = 1.0 / math.sqrt(2.0)

_NOISE_FIELDS = ("single_qubit_depolarizing", "two_qubit_depolarizing")


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
# 密度矩阵 ρ 按 ``rho[i + j * size] = ⟨i|ρ|j⟩`` 展平存储，等价于把行
# 下标放在低 n 位、列下标放在高 n 位的 2n 比特向量：左乘 U 即对低 n
# 位施加 U，右乘 U† 即对高 n 位施加 U*（x/h/cx 为实矩阵，rx/rz 的
# 共轭即角度取反），因此可以直接复用状态向量的门作用函数。

def _depolarize(rho: list[complex], n: int, qubits: tuple[int, ...], p: float) -> None:
    """就地施加 ``D_S(ρ) = (1-p)ρ + p (I_S/2^|S| ⊗ Tr_S ρ)``，S 为 qubits。"""
    size = 1 << n
    smask = 0
    for q in qubits:
        smask |= 1 << q
    d = 1 << len(qubits)
    # 子系统比特模式 s 展开到其在全系统下标中的位掩码。
    offsets = []
    for s in range(d):
        off = 0
        for k, q in enumerate(qubits):
            if (s >> k) & 1:
                off |= 1 << q
        offsets.append(off)
    keep = [i for i in range(size) if not i & smask]

    # 部分迹 M[a, b] = Σ_s ρ[a|s, b|s]，a/b 为 S 位清零的下标。
    traced: dict[int, complex] = {}
    for a in keep:
        for b in keep:
            total = 0j
            for off in offsets:
                total += rho[(a | off) + (b | off) * size]
            traced[a * size + b] = total

    inv = 1.0 - p
    for idx in range(len(rho)):
        rho[idx] *= inv
    factor = p / d
    for a in keep:
        for b in keep:
            m = traced[a * size + b]
            if m == 0j:
                continue
            add = factor * m
            for off in offsets:
                rho[(a | off) + (b | off) * size] += add


def _simulate_noisy(circuit: dict, p1: float, p2: float) -> list[complex]:
    """执行无参数电路并逐门施加退极化通道，返回末态密度矩阵（展平）。"""
    n = circuit["qubit_count"]
    size = 1 << n
    dim = size * size
    rho = [0j] * dim
    rho[0] = 1 + 0j
    for op in circuit["operations"]:
        gate = op["gate"]
        if gate == "cx":
            control = op["control"]
            target = op["target"]
            _apply_cx(rho, dim, control, target)
            _apply_cx(rho, dim, n + control, n + target)
            if p2:
                _depolarize(rho, n, (control, target), p2)
            continue

        target = op["target"]
        if gate == "x":
            _apply_x(rho, dim, target)
            _apply_x(rho, dim, n + target)
        elif gate == "h":
            _apply_h(rho, dim, target)
            _apply_h(rho, dim, n + target)
        elif gate == "rx":
            _apply_rx(rho, dim, target, op["angle"])
            _apply_rx(rho, dim, n + target, -op["angle"])
        else:  # rz
            _apply_rz(rho, dim, target, op["angle"])
            _apply_rz(rho, dim, n + target, -op["angle"])
        if p1:
            _depolarize(rho, n, (target,), p1)
    return rho


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


def _pauli_expectation_density(rho: list[complex], n: int, observable: str) -> float:
    """计算 ``Tr(Pρ)``，P 由 Pauli 串给出，返回实部并规约到 [-1, 1]。"""
    size = 1 << n
    flip = 0
    for q, ch in enumerate(observable):
        if ch in ("X", "Y"):
            flip |= 1 << q

    # Tr(Pρ) = Σ_c phase(c) ⟨c|ρ|c^flip⟩，phase 约定与状态向量版本一致。
    total = 0j
    for c in range(size):
        phase = 1 + 0j
        for q, ch in enumerate(observable):
            bit = (c >> q) & 1
            if ch == "Y":
                phase *= -1j if bit else 1j
            elif ch == "Z":
                if bit:
                    phase = -phase
        total += phase * rho[c + (c ^ flip) * size]

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


def _validate_noise(noise: Any) -> tuple[float, float] | None:
    """校验 noise 模型，返回 ``(单比特概率, 双比特概率)``。

    ``None``、空对象或概率全零时返回 None，表示走无噪声路径。
    先按单比特、双比特顺序校验已提供概率，最后拒绝未知字段。
    """
    if noise is None:
        return None
    if not isinstance(noise, dict):
        raise SimulationError("invalid_noise_model", "noise", "noise must be an object")
    for key in noise:
        if not isinstance(key, str):
            raise SimulationError("invalid_noise_model", "noise", "noise field names must be strings")

    probabilities: dict[str, float] = {}
    for field in _NOISE_FIELDS:
        if field not in noise:
            continue
        value = noise[field]
        path = f"noise.{field}"
        if not _is_number(value) or not math.isfinite(float(value)):
            raise SimulationError(
                "invalid_noise_model", path,
                f"probability at {path} must be a finite number in [0, 1]",
            )
        probability = _canon_float(float(value))
        if probability < 0.0 or probability > 1.0:
            raise SimulationError(
                "invalid_noise_model", path,
                f"probability at {path} must be a finite number in [0, 1]",
            )
        probabilities[field] = probability

    unknown = sorted(key for key in noise if key not in _NOISE_FIELDS)
    if unknown:
        field = unknown[0]
        raise SimulationError(
            "invalid_noise_model", f"noise.{field}",
            f"unknown noise field {field!r}",
        )

    p1 = probabilities.get("single_qubit_depolarizing", 0.0)
    p2 = probabilities.get("two_qubit_depolarizing", 0.0)
    if p1 == 0.0 and p2 == 0.0:
        return None
    return (p1, p2)


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
    否则每项独立采样并返回 ``{"positive", "negative"}`` 计数。可选的
    ``noise`` 为逐门局部退极化噪声模型，省略、为 None、为空对象或
    概率全零时与无噪声结果完全一致；任一概率非零时改用密度矩阵
    仿真，qubit 上限为 10。返回 ``{"qubit_count", "shots", "results"}``，
    其中 results 按 observables 输入顺序排列，全部为 JSON 原生类型。
    """
    bound = bind_parameters(circuit, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    pauli_strings = _validate_observables(observables, qubit_count)
    shot_count = _validate_shots(shots)
    resolved_seed = _validate_seed(seed, shot_count)
    resolved_noise = _validate_noise(noise)

    limit = _MAX_QUBITS if resolved_noise is None else _MAX_QUBITS_NOISY
    if qubit_count > limit:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {limit}-qubit simulation limit",
        )

    if resolved_noise is None:
        state = _simulate(bound)
        exact = [_pauli_expectation(state, obs) for obs in pauli_strings]
    else:
        rho = _simulate_noisy(bound, *resolved_noise)
        exact = [_pauli_expectation_density(rho, qubit_count, obs) for obs in pauli_strings]

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


def _simulate_with_shift(
    bound: dict,
    op_index: int,
    shift: float,
    noise: tuple[float, float] | None = None,
) -> list[complex]:
    """在绑定电路基础上把第 op_index 个旋转门角度平移 shift 后仿真。"""
    operations = [dict(op) for op in bound["operations"]]
    operations[op_index]["angle"] = operations[op_index]["angle"] + shift
    shifted = {
        "qubit_count": bound["qubit_count"],
        "parameters": [],
        "operations": operations,
    }
    if noise is None:
        return _simulate(shifted)
    return _simulate_noisy(shifted, *noise)


def estimate_gradient(circuit: Any, observables: Any, values: Any = None, noise: Any = None) -> dict:
    """精确参数移位梯度：各 observable 对全部声明参数的导数，不修改输入。

    校验与失败语义和精确 :func:`estimate_expectation` 一致。角度为
    ``aθ+b`` 的旋转门对参数 θ 的单次贡献为 ``a/2`` 乘以该门角度单独
    增减 ``π/2`` 后两次精确期望值之差；同一参数出现在多个旋转门时
    累加各门贡献，未用于参数化旋转的参数导数为 ``0.0``。可选的
    ``noise`` 与 :func:`estimate_expectation` 同义；含噪时基准期望值
    与移位期望值都来自含噪密度矩阵（噪声通道与参数无关，参数移位
    规则保持精确）。

    返回 ``{"qubit_count", "parameters", "results"}``：parameters 保持
    声明顺序；results 保持 observables 输入顺序（含重复项），每项为
    ``{"observable", "expectation", "gradients"}``，gradients 按
    parameters 顺序给出浮点导数；所有输出把 ``-0.0`` 规范为 ``0.0``。
    """
    normalized = normalize_circuit(circuit)
    bound = bind_parameters(normalized, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    pauli_strings = _validate_observables(observables, qubit_count)
    resolved_noise = _validate_noise(noise)

    limit = _MAX_QUBITS if resolved_noise is None else _MAX_QUBITS_NOISY
    if qubit_count > limit:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {limit}-qubit simulation limit",
        )

    parameters = list(normalized["parameters"])

    if resolved_noise is None:
        base_state = _simulate(bound)
        base = [_pauli_expectation(base_state, obs) for obs in pauli_strings]
    else:
        base_rho = _simulate_noisy(bound, *resolved_noise)
        base = [_pauli_expectation_density(base_rho, qubit_count, obs) for obs in pauli_strings]

    # 绑定前后操作一一对应；收集各参数出现的旋转门（下标与线性系数）。
    contributions: dict[str, list[tuple[int, float]]] = {name: [] for name in parameters}
    for i, op in enumerate(normalized["operations"]):
        angle = op.get("angle")
        if isinstance(angle, dict):
            contributions[angle["parameter"]].append((i, angle["coefficient"]))

    totals = {name: [0.0] * len(pauli_strings) for name in parameters}
    for name in parameters:
        for op_index, coefficient in contributions[name]:
            plus_state = _simulate_with_shift(bound, op_index, _HALF_PI, resolved_noise)
            minus_state = _simulate_with_shift(bound, op_index, -_HALF_PI, resolved_noise)
            factor = coefficient / 2.0
            for k, obs in enumerate(pauli_strings):
                if resolved_noise is None:
                    plus = _pauli_expectation(plus_state, obs)
                    minus = _pauli_expectation(minus_state, obs)
                else:
                    plus = _pauli_expectation_density(plus_state, qubit_count, obs)
                    minus = _pauli_expectation_density(minus_state, qubit_count, obs)
                totals[name][k] += factor * (plus - minus)

    results: list[dict] = []
    for k, obs in enumerate(pauli_strings):
        gradients = {name: _canon_float(totals[name][k]) for name in parameters}
        results.append({"observable": obs, "expectation": base[k], "gradients": gradients})

    return {"qubit_count": qubit_count, "parameters": parameters, "results": results}
