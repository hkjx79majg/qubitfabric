"""Hamiltonian 分组联合采样：共享总采样预算的分组期望值估计。

:func:`grouped_hamiltonian_expectation` 按 terms 输入顺序把各项放入最早
的兼容组，无兼容组时新建；两个 Pauli 串仅当每个量子位上的字符相同或
至少一方为 ``I`` 时兼容。组的 basis 是各位唯一的非 ``I`` 字符，无非
``I`` 字符的位置保留 ``I``。``shots`` 是所有组共享的总采样预算，按
整除结果均分，余数依组顺序各加一；``shots`` 小于分组数时抛
``insufficient_shots``。

每组只生成一批联合测量样本：末态（含噪时为密度矩阵）先旋转到组的
合并基，再采样一批 bitstring，同组各项从相同 bitstring 计算 ±1
本征值。各组随机流由 ``seed`` 与组序号派生
（``f"{seed}:{group_index}"``），相同规范化输入与 seed 的结果逐值
一致。纯 ``I`` 项本征值恒为 +1。

校验顺序：电路 → 参数绑定 → terms 结构 → observable 内容 → noise →
shots（含预算是否足够）→ seed → 状态空间上限；除新增的
``insufficient_shots`` 外复用现有异常类型、code 与 path。不修改输入，
输出仅含 JSON 原生类型。
"""

from __future__ import annotations

import bisect
import math
import random
from typing import Any

from .circuit import bind_parameters
from .optimize import _validate_terms
from .simulate import (
    _MAX_QUBITS,
    _MAX_QUBITS_NOISY,
    SimulationError,
    _canon_float,
    _dm_apply_single,
    _is_int,
    _simulate,
    _simulate_noisy,
    _validate_noise,
    _validate_observables,
    _validate_seed,
)

__all__ = ["grouped_hamiltonian_expectation"]

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def _validate_total_shots(shots: Any) -> int:
    """总采样预算：必填，排除 bool 的正整数。"""
    if not _is_int(shots) or shots <= 0:
        raise SimulationError("invalid_shots", "shots", "shots must be a positive integer")
    return shots


# ---------------------------------------------------------------------------
# 分组与预算分配
# ---------------------------------------------------------------------------

def _compatible(observable: str, basis: list[str]) -> bool:
    """observable 与组合并基逐位兼容：字符相同或至少一方为 I。"""
    return all(o == b or o == "I" or b == "I" for o, b in zip(observable, basis))


def _group_terms(observables: list[str]) -> list[dict]:
    """按输入顺序把各项放入最早的兼容组，无兼容组时新建。"""
    groups: list[dict] = []
    for index, observable in enumerate(observables):
        for group in groups:
            if _compatible(observable, group["basis"]):
                basis = group["basis"]
                for q, ch in enumerate(observable):
                    if ch != "I":
                        basis[q] = ch
                group["term_indexes"].append(index)
                break
        else:
            groups.append({"basis": list(observable), "term_indexes": [index]})
    return groups


def _allocate_shots(shots: int, group_count: int) -> list[int]:
    """总预算按整除均分，余数依组顺序各加一；预算不足抛 insufficient_shots。"""
    if shots < group_count:
        raise SimulationError(
            "insufficient_shots", "shots",
            f"shots {shots} is smaller than the {group_count} measurement groups",
        )
    base, remainder = divmod(shots, group_count)
    return [base + (1 if i < remainder else 0) for i in range(group_count)]


def _pauli_mask(observable: str) -> int:
    """非 I 位置的位掩码；纯 I 项掩码为 0，本征值恒为 +1。"""
    mask = 0
    for q, ch in enumerate(observable):
        if ch != "I":
            mask |= 1 << q
    return mask


# ---------------------------------------------------------------------------
# 基旋转与联合采样
# ---------------------------------------------------------------------------

def _rotate_state_to_basis(state: list[complex], size: int, q: int, char: str) -> None:
    """原地把量子位 q 旋转到 char 指定的测量基（X 用 H，Y 用 H·S†）。"""
    step = 1 << q
    for base in range(0, size, 2 * step):
        for off in range(step):
            i = base + off
            j = i + step
            a = state[i]
            b = state[j]
            if char == "X":
                state[i] = (a + b) * _INV_SQRT2
                state[j] = (a - b) * _INV_SQRT2
            else:  # Y: U = H·S† = [[1, -i], [1, i]] / √2
                state[i] = (a - 1j * b) * _INV_SQRT2
                state[j] = (a + 1j * b) * _INV_SQRT2


def _basis_probabilities(simulated: list[complex], basis: list[str], qubit_count: int, noisy: bool) -> list[float]:
    """末态旋转到组合并基后，各 bitstring 的联合测量概率。"""
    size = 1 << qubit_count
    if noisy:
        rho = list(simulated)
        for q, ch in enumerate(basis):
            if ch == "X":
                _dm_apply_single(rho, size, q, _INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2)
            elif ch == "Y":
                _dm_apply_single(
                    rho, size, q,
                    _INV_SQRT2, -1j * _INV_SQRT2, 1j * _INV_SQRT2, _INV_SQRT2,
                )
        return [rho[i * size + i].real for i in range(size)]
    state = list(simulated)
    for q, ch in enumerate(basis):
        if ch in ("X", "Y"):
            _rotate_state_to_basis(state, size, q, ch)
    return [amp.real * amp.real + amp.imag * amp.imag for amp in state]


def _sample_group(
    probabilities: list[float],
    group_shots: int,
    term_masks: list[int],
    rng: random.Random,
) -> list[list[int]]:
    """采样一批联合 bitstring，同组各项从相同 bitstring 累计 ±1 计数。"""
    cumulative: list[float] = []
    acc = 0.0
    for p in probabilities:
        if p > 0.0:
            acc += p
        cumulative.append(acc)
    last = len(cumulative) - 1

    counts = [[0, 0] for _ in term_masks]  # 每项 [plus, minus]
    for _ in range(group_shots):
        bitstring = bisect.bisect_left(cumulative, rng.random())
        if bitstring > last:
            bitstring = last
        for k, mask in enumerate(term_masks):
            if (bitstring & mask).bit_count() & 1:
                counts[k][1] += 1
            else:
                counts[k][0] += 1
    return counts


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def grouped_hamiltonian_expectation(
    circuit: Any,
    terms: Any,
    values: Any = None,
    shots: Any = None,
    seed: Any = None,
    noise: Any = None,
) -> dict:
    """用一个总采样预算联合估计 Hamiltonian 各项期望值，不修改输入。

    返回 ``{"qubit_count", "shots", "groups", "results", "energy"}``：
    groups 按确定的创建顺序排列，每项为 ``{"basis", "shots",
    "term_indexes"}``；results 与 terms 一一对应（保留重复项），每项为
    ``{"observable", "coefficient", "expectation", "plus_count",
    "minus_count"}``，计数之和等于所属组 shots；energy 为所有
    coefficient×expectation 按输入顺序求和。
    """
    bound = bind_parameters(circuit, {} if values is None else values)
    qubit_count = bound["qubit_count"]

    observables, coefficients = _validate_terms(terms)
    pauli_strings = _validate_observables(observables, qubit_count)
    p1, p2 = _validate_noise(noise)
    shot_count = _validate_total_shots(shots)
    groups = _group_terms(pauli_strings)
    allocations = _allocate_shots(shot_count, len(groups))
    resolved_seed = _validate_seed(seed, shot_count)

    noisy = p1 != 0.0 or p2 != 0.0
    max_qubits = _MAX_QUBITS_NOISY if noisy else _MAX_QUBITS
    if qubit_count > max_qubits:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {max_qubits}-qubit simulation limit",
        )

    simulated = _simulate_noisy(bound, p1, p2) if noisy else _simulate(bound)

    plus_counts = [0] * len(pauli_strings)
    minus_counts = [0] * len(pauli_strings)
    term_group = [0] * len(pauli_strings)
    group_infos: list[dict] = []
    for group_index, group in enumerate(groups):
        probabilities = _basis_probabilities(simulated, group["basis"], qubit_count, noisy)
        rng = random.Random(f"{resolved_seed}:{group_index}")
        term_indexes = group["term_indexes"]
        masks = [_pauli_mask(pauli_strings[i]) for i in term_indexes]
        counts = _sample_group(probabilities, allocations[group_index], masks, rng)
        for k, i in enumerate(term_indexes):
            plus_counts[i], minus_counts[i] = counts[k]
            term_group[i] = group_index
        group_infos.append({
            "basis": "".join(group["basis"]),
            "shots": allocations[group_index],
            "term_indexes": list(term_indexes),
        })

    results: list[dict] = []
    energy = 0.0
    for i, observable in enumerate(pauli_strings):
        group_shots = allocations[term_group[i]]
        expectation = _canon_float((plus_counts[i] - minus_counts[i]) / group_shots)
        energy += coefficients[i] * expectation
        results.append({
            "observable": observable,
            "coefficient": coefficients[i],
            "expectation": expectation,
            "plus_count": plus_counts[i],
            "minus_count": minus_counts[i],
        })

    return {
        "qubit_count": qubit_count,
        "shots": shot_count,
        "groups": group_infos,
        "results": results,
        "energy": _canon_float(energy),
    }
