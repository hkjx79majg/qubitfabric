"""可序列化的量子电路 IR：校验、规范化与确定性等价变换。

IR 全部由 JSON 原生类型组成（dict / list / str / int / float / bool / None），
不依赖任何第三方量子 SDK。规范化后的电路形如::

    {
        "qubit_count": 2,
        "parameters": ["theta"],
        "operations": [
            {"gate": "x", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rx", "target": 0, "angle": 1.5},
            {"gate": "rz", "target": 1,
             "angle": {"parameter": "theta", "coefficient": 1.0, "offset": 0.0}},
        ],
    }

常量旋转角规范化为 JSON 数字；参数化旋转角为
``{"parameter", "coefficient", "offset"}`` 线性表达式。
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "CircuitValidationError",
    "ParameterBindingError",
    "normalize_circuit",
    "simplify_circuit",
    "bind_parameters",
]

_TWO_PI = 2.0 * math.pi

_INVOLUTIONS = ("x", "h", "cx")
_ROTATIONS = ("rx", "rz")
_KNOWN_GATES = ("x", "h", "rx", "rz", "cx")

_EXPECTED_FIELDS = {
    "x": frozenset(("gate", "target")),
    "h": frozenset(("gate", "target")),
    "cx": frozenset(("gate", "control", "target")),
    "rx": frozenset(("gate", "target", "angle")),
    "rz": frozenset(("gate", "target", "angle")),
}
_TOP_FIELDS = frozenset(("qubit_count", "parameters", "operations"))
_ANGLE_FIELDS = frozenset(("parameter", "coefficient", "offset"))


class CircuitValidationError(ValueError):
    """电路 IR 校验失败。

    ``code`` 为稳定的机器可读错误码，``path`` 指向输入中出错的位置
    （根对象为 ``"$"``，数组元素形如 ``"operations[0].target"``）。
    每次校验只报告按输入顺序遇到的首个错误。
    """

    def __init__(self, code: str, path: str, message: str | None = None) -> None:
        self.code = code
        self.path = path
        if message is None:
            message = f"{code} at {path}"
        super().__init__(message)


class ParameterBindingError(ValueError):
    """参数绑定失败。缺失参数 code 为 ``missing_parameter``，
    未知参数为 ``unknown_parameter``；同类多个名称按字典序取首个。"""

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
    """有限/无限 JSON 实数（int 或 float），排除布尔值。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _canon_float(value: float) -> float:
    """统一浮点形态：抹掉 -0.0。"""
    if value == 0.0:
        return 0.0
    return value


def _finite_float(value: Any, path: str) -> float:
    """把 JSON 实数转为有限 float；NaN/无穷/溢出统一报 non_finite_number。"""
    result = float(value)
    if not math.isfinite(result):
        raise CircuitValidationError("non_finite_number", path, f"number at {path} must be finite")
    return _canon_float(result)


# ---------------------------------------------------------------------------
# 校验与规范化
# ---------------------------------------------------------------------------

def _validate_qubit(op: dict, key: str, path: str, qubit_count: int) -> int:
    if key not in op:
        raise CircuitValidationError("missing_field", path, f"missing field {path}")
    value = op[key]
    if not _is_int(value):
        raise CircuitValidationError("invalid_type", path, f"field {path} must be an integer qubit index")
    if value < 0 or value >= qubit_count:
        raise CircuitValidationError("qubit_out_of_range", path, f"qubit index {value} out of range at {path}")
    return value


def _validate_angle(raw: Any, path: str, parameters: list[str]) -> Any:
    if _is_number(raw):
        angle = float(raw)
        if not math.isfinite(angle):
            raise CircuitValidationError("non_finite_number", path, f"angle at {path} must be finite")
        return _canon_float(angle)

    if not isinstance(raw, dict):
        raise CircuitValidationError("invalid_type", path, f"angle at {path} must be a number or parameter expression")

    for key in raw:
        if not isinstance(key, str) or key not in _ANGLE_FIELDS:
            key_path = f"{path}.{key}" if isinstance(key, str) else path
            raise CircuitValidationError("unknown_field", key_path, f"unknown angle field at {key_path}")

    if "parameter" not in raw:
        raise CircuitValidationError("missing_field", f"{path}.parameter", f"missing field {path}.parameter")
    name = raw["parameter"]
    if not isinstance(name, str):
        raise CircuitValidationError("invalid_type", f"{path}.parameter", f"field {path}.parameter must be a string")
    if name not in parameters:
        raise CircuitValidationError("unknown_parameter", f"{path}.parameter", f"undeclared parameter {name!r} at {path}.parameter")

    coefficient = 1.0
    if "coefficient" in raw:
        value = raw["coefficient"]
        if not _is_number(value):
            raise CircuitValidationError("invalid_type", f"{path}.coefficient", f"field {path}.coefficient must be a number")
        coefficient = _finite_float(value, f"{path}.coefficient")

    offset = 0.0
    if "offset" in raw:
        value = raw["offset"]
        if not _is_number(value):
            raise CircuitValidationError("invalid_type", f"{path}.offset", f"field {path}.offset must be a number")
        offset = _finite_float(value, f"{path}.offset")

    # 系数为 0 的参数表达式与常量等价，规范化为数字形式。
    if coefficient == 0.0:
        return _canon_float(offset)
    return {"parameter": name, "coefficient": _canon_float(coefficient), "offset": _canon_float(offset)}


def _validate_operation(raw: Any, index: int, qubit_count: int, parameters: list[str]) -> dict:
    path = f"operations[{index}]"
    if not isinstance(raw, dict):
        raise CircuitValidationError("invalid_type", path, f"operation at {path} must be an object")

    if "gate" not in raw:
        raise CircuitValidationError("missing_field", f"{path}.gate", f"missing field {path}.gate")
    gate = raw["gate"]
    if not isinstance(gate, str):
        raise CircuitValidationError("invalid_type", f"{path}.gate", f"field {path}.gate must be a string")
    if gate not in _KNOWN_GATES:
        raise CircuitValidationError("unknown_gate", f"{path}.gate", f"unknown gate {gate!r} at {path}.gate")

    expected = _EXPECTED_FIELDS[gate]
    for key in raw:
        if not isinstance(key, str) or key not in expected:
            key_path = f"{path}.{key}" if isinstance(key, str) else path
            raise CircuitValidationError("unknown_field", key_path, f"unknown field at {key_path}")

    if gate == "cx":
        control = _validate_qubit(raw, "control", f"{path}.control", qubit_count)
        target = _validate_qubit(raw, "target", f"{path}.target", qubit_count)
        if control == target:
            raise CircuitValidationError(
                "control_target_same", f"{path}.control", f"cx control and target must differ at {path}"
            )
        return {"gate": "cx", "control": control, "target": target}

    target = _validate_qubit(raw, "target", f"{path}.target", qubit_count)
    if gate in ("x", "h"):
        return {"gate": gate, "target": target}

    # rx / rz
    if "angle" not in raw:
        raise CircuitValidationError("missing_field", f"{path}.angle", f"missing field {path}.angle")
    angle = _validate_angle(raw["angle"], f"{path}.angle", parameters)
    return {"gate": gate, "target": target, "angle": angle}


def normalize_circuit(data: Any) -> dict:
    """校验 JSON 兼容的电路描述并返回规范化的新对象，不修改输入。

    省略的 ``parameters`` / ``operations`` 补为空列表；旋转角统一为
    float 或完整的参数线性表达式；相同语义的输入得到字段形式一致的结果。
    """
    if not isinstance(data, dict):
        raise CircuitValidationError("invalid_type", "$", "circuit must be a JSON object")

    # 先按输入（字典插入）顺序拒绝未知字段。
    for key in data:
        if not isinstance(key, str) or key not in _TOP_FIELDS:
            path = key if isinstance(key, str) else "$"
            raise CircuitValidationError("unknown_field", path, f"unknown top-level field {path!r}")

    if "qubit_count" not in data:
        raise CircuitValidationError("missing_field", "qubit_count", "missing field qubit_count")
    qubit_count = data["qubit_count"]
    if not _is_int(qubit_count):
        raise CircuitValidationError("invalid_type", "qubit_count", "qubit_count must be a non-negative integer")
    if qubit_count < 0:
        raise CircuitValidationError("invalid_value", "qubit_count", "qubit_count must be non-negative")

    parameters: list[str] = []
    if "parameters" in data:
        raw_parameters = data["parameters"]
        if not isinstance(raw_parameters, list):
            raise CircuitValidationError("invalid_type", "parameters", "parameters must be an array of strings")
        seen: set[str] = set()
        for i, name in enumerate(raw_parameters):
            path = f"parameters[{i}]"
            if not isinstance(name, str):
                raise CircuitValidationError("invalid_type", path, f"parameter name at {path} must be a string")
            if name == "":
                raise CircuitValidationError("invalid_parameter_name", path, f"parameter name at {path} must be non-empty")
            if name in seen:
                raise CircuitValidationError("duplicate_parameter", path, f"duplicate parameter name {name!r} at {path}")
            seen.add(name)
            parameters.append(name)

    operations: list[dict] = []
    if "operations" in data:
        raw_operations = data["operations"]
        if not isinstance(raw_operations, list):
            raise CircuitValidationError("invalid_type", "operations", "operations must be an array")
        for i, raw_op in enumerate(raw_operations):
            operations.append(_validate_operation(raw_op, i, qubit_count, parameters))

    return {"qubit_count": qubit_count, "parameters": parameters, "operations": operations}


# ---------------------------------------------------------------------------
# simplify：确定性等价变换
# ---------------------------------------------------------------------------

def _wrap_angle(value: float) -> float:
    """规约到半开区间 [-pi, pi)。"""
    wrapped = (value + math.pi) % _TWO_PI - math.pi
    return _canon_float(wrapped)


def _touches(operation: dict, qubit: int) -> bool:
    if operation["gate"] == "cx":
        return operation["control"] == qubit or operation["target"] == qubit
    return operation["target"] == qubit


def _same_involution(left: dict, right: dict) -> bool:
    """两个相邻的 x/h/cx 作用位是否完全相同（调用时 gate 已一致）。"""
    if left["gate"] != right["gate"]:
        return False
    if left["target"] != right["target"]:
        return False
    if left["gate"] == "cx":
        return left["control"] == right["control"]
    return True


def _add_angles(left: Any, right: Any) -> Any:
    """合并两个同轴旋转角。

    常量与常量、参数与同参数表达式可精确合并；不同参数的表达式返回 None，
    调用方据此保持原顺序。返回的常量不做零判定，由调用方统一规约。
    """
    if isinstance(left, float) and isinstance(right, float):
        return _canon_float(left + right)

    if isinstance(left, float):
        return {
            "parameter": right["parameter"],
            "coefficient": _canon_float(right["coefficient"]),
            "offset": _canon_float(right["offset"] + left),
        }
    if isinstance(right, float):
        return {
            "parameter": left["parameter"],
            "coefficient": _canon_float(left["coefficient"]),
            "offset": _canon_float(left["offset"] + right),
        }

    if left["parameter"] != right["parameter"]:
        return None
    coefficient = _canon_float(left["coefficient"] + right["coefficient"])
    offset = _canon_float(left["offset"] + right["offset"])
    if coefficient == 0.0:
        return _canon_float(offset)
    return {"parameter": left["parameter"], "coefficient": coefficient, "offset": offset}


def _canonicalize_angle(angle: Any) -> Any:
    """simplify 输出角度的统一形态：常量包裹到 [-pi, pi)，参数表达式折叠常量偏移。"""
    if isinstance(angle, float):
        return _wrap_angle(angle)
    coefficient = angle["coefficient"]
    if coefficient == 0.0:
        return _wrap_angle(angle["offset"])
    return {
        "parameter": angle["parameter"],
        "coefficient": _canon_float(coefficient),
        "offset": _wrap_angle(angle["offset"]),
    }


def simplify_circuit(circuit: Any) -> dict:
    """对规范化电路做确定性等价变换。

    - 相邻且作用位相同的两个 x/h/cx 相互抵消；
    - 同轴、同目标位且中间没有作用于该位的操作的 rx/rz 合并；
    - 常量角规约到 [-pi, pi)，规约为零则删除；
    - 参数化旋转仅在线性表达式可精确合并时处理，否则保持原顺序。

    返回 ``{"circuit", "removed_operations", "merged_operations"}``，
    其中 circuit 为全新的合法规范 IR；重复简化结果不变。
    """
    normalized = normalize_circuit(circuit)

    kept: list[dict] = []
    removed = 0
    merged = 0

    for op in normalized["operations"]:
        gate = op["gate"]

        if gate in _INVOLUTIONS:
            if kept and _same_involution(kept[-1], op):
                kept.pop()
                removed += 2
            else:
                kept.append(dict(op))
            continue

        # rx / rz
        angle: Any = op["angle"]
        if isinstance(angle, float):
            angle = _wrap_angle(angle)
            if angle == 0.0:
                removed += 1
                continue

        # 向前找最近一个作用于目标位的操作：只有同轴旋转可以合并，
        # 其间的操作若都不触碰目标位，则旋转与之交换、合并安全。
        merge_index: int | None = None
        for j in range(len(kept) - 1, -1, -1):
            if _touches(kept[j], op["target"]):
                if kept[j]["gate"] == gate:
                    merge_index = j
                break

        if merge_index is not None:
            combined = _add_angles(kept[merge_index]["angle"], angle)
            if combined is not None:
                merged += 1
                if isinstance(combined, float) and _wrap_angle(combined) == 0.0:
                    del kept[merge_index]
                    removed += 1
                else:
                    kept[merge_index]["angle"] = _canonicalize_angle(combined)
                continue

        kept.append({"gate": gate, "target": op["target"], "angle": _canonicalize_angle(angle)})

    result = {
        "qubit_count": normalized["qubit_count"],
        "parameters": list(normalized["parameters"]),
        "operations": kept,
    }
    return {"circuit": result, "removed_operations": removed, "merged_operations": merged}


# ---------------------------------------------------------------------------
# bind：参数绑定
# ---------------------------------------------------------------------------

def bind_parameters(circuit: Any, values: Any) -> dict:
    """用 ``values`` 映射把电路参数替换为有限实数角度，返回全新电路。

    全部参数必须且只能绑定一次：缺失参数抛 ``missing_parameter``，
    未知绑定抛 ``unknown_parameter``（同类多个名称按字典序取首个）。
    成功后 ``parameters`` 为空且不存在参数化角度；输入电路与映射均不修改。
    """
    normalized = normalize_circuit(circuit)
    declared = normalized["parameters"]

    if not isinstance(values, dict):
        raise ParameterBindingError("invalid_type", "$", "parameter bindings must be an object")

    for key in values:
        if not isinstance(key, str):
            raise ParameterBindingError("invalid_type", "$", "parameter names must be strings")

    # 绑定值本身的类型/数值错误按映射插入顺序报告。
    for name, value in values.items():
        if not _is_number(value):
            raise ParameterBindingError("invalid_type", name, f"binding for {name!r} must be a finite number")
        if not math.isfinite(float(value)):
            raise ParameterBindingError("non_finite_number", name, f"binding for {name!r} must be finite")

    missing = sorted(name for name in declared if name not in values)
    if missing:
        name = missing[0]
        raise ParameterBindingError("missing_parameter", name, f"missing binding for parameter {name!r}")

    unknown = sorted(name for name in values if name not in declared)
    if unknown:
        name = unknown[0]
        raise ParameterBindingError("unknown_parameter", name, f"unknown parameter binding {name!r}")

    operations: list[dict] = []
    for i, op in enumerate(normalized["operations"]):
        if op["gate"] in _INVOLUTIONS:
            operations.append(dict(op))
            continue

        angle = op["angle"]
        if isinstance(angle, float):
            bound_angle = _canon_float(angle)
        else:
            name = angle["parameter"]
            bound_angle = angle["coefficient"] * float(values[name]) + angle["offset"]
            if not math.isfinite(bound_angle):
                raise ParameterBindingError(
                    "non_finite_number",
                    f"operations[{i}].angle",
                    f"bound angle at operations[{i}].angle is not finite",
                )
            bound_angle = _canon_float(bound_angle)
        operations.append({"gate": op["gate"], "target": op["target"], "angle": bound_angle})

    return {
        "qubit_count": normalized["qubit_count"],
        "parameters": [],
        "operations": operations,
    }
