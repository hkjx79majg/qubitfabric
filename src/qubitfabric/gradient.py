"""精确参数移位梯度：变分参数对 Pauli 期望值的导数。

仅覆盖 rx/rz 的线性参数角 ``aθ+b`` 与无噪声精确计算。角度为 ``aθ+b``
的旋转门对参数 θ 的单次贡献定义为 ``a/2`` 乘以该门角度单独增加 ``π/2``
与单独减少 ``π/2`` 后两次精确期望值之差，其他门保持绑定值；同一参数
出现在多个旋转门时累加各门贡献，未用于有效参数化旋转的参数导数为
``0.0``。

校验与失败语义和 :func:`qubitfabric.simulate.estimate_expectation` 完全
一致：电路校验、参数绑定、observables 校验、状态空间上限按相同顺序
执行并抛出相同错误类型与 code/path。
"""

from __future__ import annotations

import math
from typing import Any

from .circuit import bind_parameters, normalize_circuit
from .simulate import (
    _MAX_QUBITS,
    SimulationError,
    _canon_float,
    _pauli_expectation,
    _simulate,
    _validate_observables,
)

__all__ = ["estimate_gradient"]

_HALF_PI = math.pi / 2.0
_ROTATION_GATES = ("rx", "rz")


def _shifted_circuit(bound: dict, index: int, delta: float) -> dict:
    """返回只有第 ``index`` 个操作的绑定角度偏移 ``delta`` 的新电路。"""
    operations = []
    for j, op in enumerate(bound["operations"]):
        if j == index:
            op = dict(op)
            op["angle"] = op["angle"] + delta
        operations.append(op)
    return {"qubit_count": bound["qubit_count"], "parameters": [], "operations": operations}


def estimate_gradient(circuit: Any, observables: Any, values: Any = None) -> dict:
    """计算各 observable 期望值对全部声明参数的精确参数移位梯度。

    ``circuit`` / ``observables`` / ``values`` 的含义与
    :func:`qubitfabric.simulate.estimate_expectation` 一致；无参数电路
    允许省略 ``values`` 或传空映射。返回
    ``{"qubit_count", "parameters", "results"}``：``parameters`` 保持声明
    顺序，``results`` 保持 observable 输入顺序并保留重复项，每项包含
    ``observable``、当前绑定点的精确 ``expectation`` 和按 ``parameters``
    顺序给出的浮点 ``gradients`` 映射。所有输出把 ``-0.0`` 规范为
    ``0.0``，且不修改任何输入。
    """
    bound = bind_parameters(circuit, {} if values is None else values)
    normalized = normalize_circuit(circuit)
    qubit_count = bound["qubit_count"]

    pauli_strings = _validate_observables(observables, qubit_count)

    if qubit_count > _MAX_QUBITS:
        raise SimulationError(
            "state_space_too_large", "qubit_count",
            f"qubit_count {qubit_count} exceeds the {_MAX_QUBITS}-qubit state vector limit",
        )

    parameters: list[str] = normalized["parameters"]
    parameter_index = {name: i for i, name in enumerate(parameters)}

    base_state = _simulate(bound)
    base = [_pauli_expectation(base_state, obs) for obs in pauli_strings]

    # gradients[参数下标][observable 下标]，按门累加参数移位贡献。
    gradients = [[0.0] * len(pauli_strings) for _ in parameters]
    for i, op in enumerate(normalized["operations"]):
        if op["gate"] not in _ROTATION_GATES:
            continue
        angle = op["angle"]
        if not isinstance(angle, dict):
            continue
        p = parameter_index[angle["parameter"]]
        half_coefficient = angle["coefficient"] / 2.0

        plus_state = _simulate(_shifted_circuit(bound, i, _HALF_PI))
        minus_state = _simulate(_shifted_circuit(bound, i, -_HALF_PI))
        for k, obs in enumerate(pauli_strings):
            plus = _pauli_expectation(plus_state, obs)
            minus = _pauli_expectation(minus_state, obs)
            gradients[p][k] += half_coefficient * (plus - minus)

    results = []
    for k, obs in enumerate(pauli_strings):
        results.append({
            "observable": obs,
            "expectation": base[k],
            "gradients": {
                name: _canon_float(gradients[p][k])
                for p, name in enumerate(parameters)
            },
        })

    return {"qubit_count": qubit_count, "parameters": list(parameters), "results": results}
