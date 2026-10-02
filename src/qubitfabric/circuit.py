"""Serializable quantum-circuit IR, validation, and equivalence transforms.

The IR is composed of JSON-native types only (dict / list / str / int /
float / bool / None) and does not depend on any third-party quantum SDK.

Canonical circuit form::

    {
        "qubit_count": 2,                 # non-negative int, required
        "parameters": ["theta"],          # declared parameter names, default []
        "operations": [                   # default []
            {"gate": "x", "target": 0},
            {"gate": "h", "target": 1},
            {"gate": "rx", "target": 0, "angle": 1.5},
            {"gate": "rz", "target": 0,
             "angle": {"parameter": "theta", "coefficient": 1.0, "offset": 0.0}},
            {"gate": "cx", "control": 0, "target": 1},
        ],
    }

Angles are either finite real numbers (normalized to ``float``) or linear
parameter references ``coefficient * parameter + offset`` with finite real
coefficient and offset (defaults 1.0 / 0.0). A bare string angle is shorthand
for ``{"parameter": name, "coefficient": 1.0, "offset": 0.0}``.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "CircuitValidationError",
    "ParameterBindingError",
    "normalize_circuit",
    "simplify_circuit",
    "bind_circuit",
]

_TWO_PI = 2.0 * math.pi

_SELF_INVERSE_GATES = ("x", "h", "cx")
_ROTATION_GATES = ("rx", "rz")

# Required fields per gate, in canonical output order (after "gate").
_GATE_FIELDS = {
    "x": ("target",),
    "h": ("target",),
    "rx": ("target", "angle"),
    "rz": ("target", "angle"),
    "cx": ("control", "target"),
}


class CircuitValidationError(ValueError):
    """Raised when a circuit specification (or bindings payload) is invalid.

    Carries a stable machine-readable ``code`` and a ``path`` (list of
    object keys / list indices, JSON-native) pointing at the offending
    location. Only the first error encountered in input order is reported.
    """

    def __init__(self, code: str, path: list, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = list(path)
        self.message = message

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ParameterBindingError(ValueError):
    """Raised when parameter bindings do not match the declared parameters.

    ``code`` is ``"missing_parameter"`` when a declared parameter has no
    binding, or ``"unknown_parameter"`` when a binding names an undeclared
    parameter. ``parameter`` holds the offending name; when several names
    qualify, the lexicographically first one is reported.
    """

    def __init__(self, code: str, path: list, parameter: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = list(path)
        self.parameter = parameter
        self.message = message

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def _fail(code: str, path: list, message: str) -> None:
    raise CircuitValidationError(code, path, message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(value: Any, path: list) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("invalid_type", path, "expected a finite real number")
    result = float(value)
    if not math.isfinite(result):
        _fail("non_finite_number", path, "number must be finite (no NaN or infinity)")
    return result


def _reduce_angle(value: float) -> float:
    """Reduce a constant angle to the half-open interval [-pi, pi)."""
    return (value + math.pi) % _TWO_PI - math.pi


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def normalize_circuit(spec: Any) -> dict:
    """Validate a JSON-compatible circuit spec and return its canonical IR.

    The input is never modified. Omitted fields are filled with defaults and
    semantically equal inputs produce identical field forms.
    """
    if not isinstance(spec, dict):
        _fail("invalid_type", [], "circuit spec must be a JSON object")
    allowed = {"qubit_count", "parameters", "operations"}
    for key in spec:
        if key not in allowed:
            _fail("unknown_field", [key], f"unknown field {key!r}")
    if "qubit_count" not in spec:
        _fail("missing_field", ["qubit_count"], "missing required field 'qubit_count'")
    qubit_count = _normalize_qubit_count(spec["qubit_count"])
    parameters = _normalize_parameters(spec.get("parameters", []))
    operations = _normalize_operations(
        spec.get("operations", []), qubit_count, frozenset(parameters)
    )
    return {
        "qubit_count": qubit_count,
        "parameters": parameters,
        "operations": operations,
    }


def _normalize_qubit_count(value: Any) -> int:
    if not _is_int(value):
        _fail("invalid_type", ["qubit_count"], "qubit_count must be a non-negative integer")
    if value < 0:
        _fail("invalid_value", ["qubit_count"], "qubit_count must be non-negative")
    return value


def _normalize_parameters(value: Any) -> list:
    if not isinstance(value, list):
        _fail("invalid_type", ["parameters"], "parameters must be a list of names")
    seen: set[str] = set()
    result: list[str] = []
    for index, name in enumerate(value):
        path = ["parameters", index]
        if not isinstance(name, str):
            _fail("invalid_type", path, "parameter name must be a string")
        if name == "":
            _fail("invalid_parameter_name", path, "parameter name must be non-empty")
        if name in seen:
            _fail("duplicate_parameter", path, f"duplicate parameter {name!r}")
        seen.add(name)
        result.append(name)
    return result


def _normalize_operations(value: Any, qubit_count: int, parameters: frozenset) -> list:
    if not isinstance(value, list):
        _fail("invalid_type", ["operations"], "operations must be a list")
    return [
        _normalize_operation(op, ["operations", index], qubit_count, parameters)
        for index, op in enumerate(value)
    ]


def _normalize_operation(op: Any, path: list, qubit_count: int, parameters: frozenset) -> dict:
    if not isinstance(op, dict):
        _fail("invalid_type", path, "operation must be a JSON object")
    if "gate" not in op:
        _fail("missing_field", path + ["gate"], "operation is missing 'gate'")
    gate = op["gate"]
    if not isinstance(gate, str):
        _fail("invalid_type", path + ["gate"], "gate must be a string")
    if gate not in _GATE_FIELDS:
        _fail("unknown_gate", path + ["gate"], f"unknown gate {gate!r}")
    required = _GATE_FIELDS[gate]
    allowed = set(required) | {"gate"}
    normalized: dict[str, Any] = {}
    for key in op:
        if key == "gate":
            continue
        if key not in allowed:
            _fail("unknown_field", path + [key], f"field {key!r} is not allowed for gate {gate!r}")
        value = op[key]
        if key in ("target", "control"):
            normalized[key] = _normalize_qubit_index(value, path + [key], qubit_count)
        else:  # "angle"
            normalized[key] = _normalize_angle(value, path + [key], parameters)
    for field in required:
        if field not in normalized:
            _fail("missing_field", path + [field], f"gate {gate!r} is missing {field!r}")
    if gate == "cx" and normalized["control"] == normalized["target"]:
        _fail("same_qubits", path + ["target"], "cx control and target must differ")
    if gate in ("rx", "rz"):
        return {"gate": gate, "target": normalized["target"], "angle": normalized["angle"]}
    if gate == "cx":
        return {"gate": "cx", "control": normalized["control"], "target": normalized["target"]}
    return {"gate": gate, "target": normalized["target"]}


def _normalize_qubit_index(value: Any, path: list, qubit_count: int) -> int:
    if not _is_int(value):
        _fail("invalid_type", path, "qubit index must be an integer")
    if value < 0 or value >= qubit_count:
        _fail(
            "qubit_out_of_range",
            path,
            f"qubit index {value} out of range for {qubit_count} qubit(s)",
        )
    return value


def _normalize_angle(value: Any, path: list, parameters: frozenset) -> Any:
    if isinstance(value, bool):
        _fail("invalid_type", path, "angle must be a number, parameter name, or object")
    if isinstance(value, (int, float)):
        return _finite_float(value, path)
    if isinstance(value, str):
        if value not in parameters:
            _fail("undefined_parameter", path, f"angle references undeclared parameter {value!r}")
        return {"parameter": value, "coefficient": 1.0, "offset": 0.0}
    if isinstance(value, dict):
        allowed = {"parameter", "coefficient", "offset"}
        for key in value:
            if key not in allowed:
                _fail("unknown_field", path + [key], f"unknown angle field {key!r}")
        if "parameter" not in value:
            _fail("missing_field", path + ["parameter"], "parameterized angle is missing 'parameter'")
        name = value["parameter"]
        if not isinstance(name, str):
            _fail("invalid_type", path + ["parameter"], "parameter must be a string")
        if name not in parameters:
            _fail(
                "undefined_parameter",
                path + ["parameter"],
                f"angle references undeclared parameter {name!r}",
            )
        coefficient = _finite_float(value.get("coefficient", 1.0), path + ["coefficient"])
        offset = _finite_float(value.get("offset", 0.0), path + ["offset"])
        return {"parameter": name, "coefficient": coefficient, "offset": offset}
    _fail("invalid_type", path, "angle must be a number, parameter name, or object")


# ---------------------------------------------------------------------------
# simplify
# ---------------------------------------------------------------------------

def simplify_circuit(circuit: Any) -> dict:
    """Apply deterministic equivalence transforms to a circuit.

    Returns ``{"circuit": <canonical IR>, "removed_operations": int,
    "merged_operations": int}``. Adjacent self-inverse pairs (x/x, h/h,
    cx/cx on the same qubits) are deleted; same-axis rx/rz rotations on the
    same qubit with no intervening same-qubit operation are merged when
    their angles combine exactly (constants, or linear expressions in the
    same parameter); constant angles are reduced to [-pi, pi) and deleted
    when they reduce to zero. The transform is idempotent: simplifying the
    result again changes nothing and reports zero counts.
    """
    normalized = normalize_circuit(circuit)
    operations = normalized["operations"]
    removed = 0
    merged = 0
    while True:
        operations, deleted = _reduce_constant_angles(operations)
        operations, cancelled = _cancel_self_inverse(operations)
        operations, combined, dropped = _merge_rotations(operations)
        removed += deleted + cancelled + dropped
        merged += combined
        if deleted + cancelled + dropped + combined == 0:
            break
    result = {
        "qubit_count": normalized["qubit_count"],
        "parameters": normalized["parameters"],
        "operations": operations,
    }
    return {
        "circuit": result,
        "removed_operations": removed,
        "merged_operations": merged,
    }


def _reduce_constant_angles(operations: list) -> tuple[list, int]:
    result = []
    removed = 0
    for op in operations:
        if op["gate"] in _ROTATION_GATES and isinstance(op["angle"], float):
            angle = _reduce_angle(op["angle"])
            if angle == 0.0:
                removed += 1
                continue
            op = {"gate": op["gate"], "target": op["target"], "angle": angle}
        result.append(op)
    return result, removed


def _cancel_self_inverse(operations: list) -> tuple[list, int]:
    stack: list[dict] = []
    removed = 0
    for op in operations:
        if op["gate"] in _SELF_INVERSE_GATES and stack and _same_self_inverse(stack[-1], op):
            stack.pop()
            removed += 2
        else:
            stack.append(op)
    return stack, removed


def _same_self_inverse(first: dict, second: dict) -> bool:
    if first["gate"] != second["gate"]:
        return False
    gate = first["gate"]
    if gate == "cx":
        return first["control"] == second["control"] and first["target"] == second["target"]
    return gate in ("x", "h") and first["target"] == second["target"]


def _merge_rotations(operations: list) -> tuple[list, int, int]:
    emitted: list[dict | None] = []
    pending: dict[int, list] = {}  # qubit -> [index in emitted, axis]
    merged = 0
    removed = 0
    for op in operations:
        gate = op["gate"]
        if gate in _ROTATION_GATES:
            qubit = op["target"]
            entry = pending.get(qubit)
            if entry is not None and entry[1] == gate:
                combined = _combine_angles(emitted[entry[0]]["angle"], op["angle"])
                if combined is not None:
                    merged += 1
                    if isinstance(combined, float) and combined == 0.0:
                        emitted[entry[0]] = None
                        removed += 1
                        del pending[qubit]
                    else:
                        emitted[entry[0]] = {"gate": gate, "target": qubit, "angle": combined}
                    continue
            emitted.append(op)
            pending[qubit] = [len(emitted) - 1, gate]
        else:
            qubits = (op["target"],) if gate in ("x", "h") else (op["control"], op["target"])
            for qubit in qubits:
                pending.pop(qubit, None)
            emitted.append(op)
    return [op for op in emitted if op is not None], merged, removed


def _as_linear(angle: Any) -> tuple[str | None, float, float]:
    """Express an angle as (parameter, coefficient, offset)."""
    if isinstance(angle, dict):
        return angle["parameter"], angle["coefficient"], angle["offset"]
    return None, 0.0, angle


def _combine_angles(first: Any, second: Any) -> Any:
    """Combine two rotation angles exactly, or return None if not possible.

    A constant angle is a linear expression with no parameter, so it merges
    with anything; two parameterized angles merge only when they reference
    the same parameter.
    """
    param_a, coeff_a, offset_a = _as_linear(first)
    param_b, coeff_b, offset_b = _as_linear(second)
    if param_a is not None and param_b is not None and param_a != param_b:
        return None
    parameter = param_a if param_a is not None else param_b
    coefficient = coeff_a + coeff_b
    offset = _reduce_angle(offset_a + offset_b)
    if coefficient == 0.0:
        return float(offset)
    return {"parameter": parameter, "coefficient": float(coefficient), "offset": float(offset)}


# ---------------------------------------------------------------------------
# bind
# ---------------------------------------------------------------------------

def bind_circuit(circuit: Any, bindings: Any) -> dict:
    """Substitute parameter bindings into a circuit and return canonical IR.

    ``bindings`` must map every declared parameter to a finite real number
    and nothing else. The result declares no parameters. The input circuit
    is never modified.
    """
    normalized = normalize_circuit(circuit)
    if not isinstance(bindings, dict):
        _fail("invalid_type", ["bindings"], "bindings must be an object mapping names to numbers")
    values: dict[str, float] = {}
    for name, value in bindings.items():
        if not isinstance(name, str):
            _fail("invalid_type", ["bindings", name], "binding names must be strings")
        values[name] = _finite_float(value, ["bindings", name])
    declared = normalized["parameters"]
    missing = sorted(set(declared) - set(values))
    if missing:
        name = missing[0]
        raise ParameterBindingError(
            "missing_parameter",
            ["parameters", name],
            name,
            f"no binding provided for parameter {name!r}",
        )
    unknown = sorted(set(values) - set(declared))
    if unknown:
        name = unknown[0]
        raise ParameterBindingError(
            "unknown_parameter",
            ["bindings", name],
            name,
            f"binding provided for undeclared parameter {name!r}",
        )
    operations = []
    for op in normalized["operations"]:
        angle = op.get("angle")
        if isinstance(angle, dict):
            value = angle["coefficient"] * values[angle["parameter"]] + angle["offset"]
            op = {"gate": op["gate"], "target": op["target"], "angle": float(value)}
        operations.append(op)
    return {
        "qubit_count": normalized["qubit_count"],
        "parameters": [],
        "operations": operations,
    }
