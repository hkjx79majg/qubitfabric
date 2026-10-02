import copy
import json
import math
import unittest

from qubitfabric.service import (
    CircuitValidationError,
    ParameterBindingError,
    Service,
)


def err(fn, *args):
    try:
        fn(*args)
    except CircuitValidationError as exc:
        return exc
    raise AssertionError("CircuitValidationError not raised")


class NormalizeTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_empty_zero_qubit_circuit(self):
        for data in ({"qubit_count": 0}, {"qubit_count": 0, "parameters": [], "operations": []}):
            with self.subTest(data=data):
                circuit = self.svc.create_circuit(data)
                self.assertEqual(circuit, {"qubit_count": 0, "parameters": [], "operations": []})

    def test_missing_qubit_count(self):
        e = err(self.svc.create_circuit, {"parameters": [], "operations": []})
        self.assertEqual((e.code, e.path), ("missing_field", "qubit_count"))

    def test_supported_gates_and_default_fields(self):
        circuit = self.svc.create_circuit({"qubit_count": 2, "operations": [
            {"gate": "x", "target": 0},
            {"gate": "h", "target": 1},
            {"gate": "rx", "target": 0, "angle": 1},
            {"gate": "rz", "target": 1, "angle": 1.5},
            {"gate": "cx", "control": 0, "target": 1},
        ]})
        self.assertEqual(circuit["parameters"], [])
        self.assertEqual(circuit["operations"][2]["angle"], 1.0)
        self.assertEqual(circuit["operations"], [
            {"gate": "x", "target": 0},
            {"gate": "h", "target": 1},
            {"gate": "rx", "target": 0, "angle": 1.0},
            {"gate": "rz", "target": 1, "angle": 1.5},
            {"gate": "cx", "control": 0, "target": 1},
        ])

    def test_parameter_angle_full_form(self):
        circuit = self.svc.create_circuit({"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}},
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta", "coefficient": -2, "offset": 0.25}},
        ]})
        self.assertEqual(circuit["operations"][0]["angle"],
                         {"parameter": "theta", "coefficient": 1.0, "offset": 0.0})
        self.assertEqual(circuit["operations"][1]["angle"],
                         {"parameter": "theta", "coefficient": -2.0, "offset": 0.25})

    def test_zero_coefficient_becomes_constant(self):
        circuit = self.svc.create_circuit({"qubit_count": 1, "parameters": ["p"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "p", "coefficient": 0, "offset": 3}},
        ]})
        self.assertEqual(circuit["operations"][0]["angle"], 3.0)

    def test_input_not_mutated_and_json_serializable(self):
        data = {"qubit_count": 1, "parameters": ["a"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "a", "coefficient": 1, "offset": 0}},
        ]}
        snapshot = copy.deepcopy(data)
        result = self.svc.create_circuit(data)
        self.assertEqual(data, snapshot)
        json.dumps(result)
        result["operations"].append({"gate": "x", "target": 0})
        again = self.svc.create_circuit(data)
        self.assertEqual(len(again["operations"]), 1)

    def test_minus_zero_normalized(self):
        circuit = self.svc.create_circuit({"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": -0.0},
        ]})
        dumped = json.dumps(circuit)
        self.assertNotIn("-0", dumped)

    # ---- validation errors ----------------------------------------------

    def test_top_level_errors(self):
        self.assertEqual(err(self.svc.create_circuit, []).code, "invalid_type")
        self.assertEqual(err(self.svc.create_circuit, []).path, "$")
        self.assertEqual(err(self.svc.create_circuit, {"operations": []}).code, "missing_field")
        self.assertEqual(err(self.svc.create_circuit, {"qubit_count": -1}).code, "invalid_value")
        self.assertEqual(err(self.svc.create_circuit, {"qubit_count": True}).code, "invalid_type")
        self.assertEqual(err(self.svc.create_circuit, {"qubit_count": 1.5}).code, "invalid_type")
        e = err(self.svc.create_circuit, {"qubit_count": 1, "bogus": 1})
        self.assertEqual(e.code, "unknown_field")
        self.assertEqual(e.path, "bogus")

    def test_parameter_errors(self):
        self.assertEqual(err(self.svc.create_circuit, {"qubit_count": 0, "parameters": "x"}).path, "parameters")
        e = err(self.svc.create_circuit, {"qubit_count": 0, "parameters": ["a", "a"]})
        self.assertEqual((e.code, e.path), ("duplicate_parameter", "parameters[1]"))
        e = err(self.svc.create_circuit, {"qubit_count": 0, "parameters": [""]})
        self.assertEqual((e.code, e.path), ("invalid_parameter_name", "parameters[0]"))
        e = err(self.svc.create_circuit, {"qubit_count": 0, "parameters": [1]})
        self.assertEqual((e.code, e.path), ("invalid_type", "parameters[0]"))

    def test_operation_errors(self):
        self.assertEqual(err(self.svc.create_circuit, {"qubit_count": 1, "operations": "x"}).code, "invalid_type")
        self.assertEqual(err(self.svc.create_circuit, {"qubit_count": 1, "operations": [1]}).path, "operations[0]")
        e = err(self.svc.create_circuit, {"qubit_count": 1, "operations": [{"target": 0}]})
        self.assertEqual((e.code, e.path), ("missing_field", "operations[0].gate"))
        e = err(self.svc.create_circuit, {"qubit_count": 1, "operations": [{"gate": "z", "target": 0}]})
        self.assertEqual((e.code, e.path), ("unknown_gate", "operations[0].gate"))
        e = err(self.svc.create_circuit, {"qubit_count": 1, "operations": [{"gate": "x", "target": 0, "q": 1}]})
        self.assertEqual((e.code, e.path), ("unknown_field", "operations[0].q"))
        e = err(self.svc.create_circuit, {"qubit_count": 1, "operations": [{"gate": "x", "target": 3}]})
        self.assertEqual((e.code, e.path), ("qubit_out_of_range", "operations[0].target"))
        e = err(self.svc.create_circuit, {"qubit_count": 1, "operations": [{"gate": "x", "target": True}]})
        self.assertEqual((e.code, e.path), ("invalid_type", "operations[0].target"))
        e = err(self.svc.create_circuit, {"qubit_count": 2, "operations": [{"gate": "cx", "control": 0, "target": 0}]})
        self.assertEqual((e.code, e.path), ("control_target_same", "operations[0].control"))
        e = err(self.svc.create_circuit, {"qubit_count": 2, "operations": [{"gate": "cx", "control": 5, "target": 0}]})
        self.assertEqual((e.code, e.path), ("qubit_out_of_range", "operations[0].control"))
        e = err(self.svc.create_circuit, {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0}]})
        self.assertEqual((e.code, e.path), ("missing_field", "operations[0].angle"))

    def test_angle_errors(self):
        base = {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": None}]}
        base["operations"][0]["angle"] = "x"
        self.assertEqual(err(self.svc.create_circuit, base).path, "operations[0].angle")
        base["operations"][0]["angle"] = True
        self.assertEqual(err(self.svc.create_circuit, base).code, "invalid_type")
        base["operations"][0]["angle"] = float("nan")
        self.assertEqual(err(self.svc.create_circuit, base).code, "non_finite_number")
        base["operations"][0]["angle"] = float("inf")
        self.assertEqual(err(self.svc.create_circuit, base).code, "non_finite_number")
        base["operations"][0]["angle"] = {"parameter": "p"}
        self.assertEqual(err(self.svc.create_circuit, base).code, "unknown_parameter")
        base["parameters"] = ["p"]
        base["operations"][0]["angle"] = {"parameter": "p", "coefficient": True}
        self.assertEqual(err(self.svc.create_circuit, base).path, "operations[0].angle.coefficient")
        base["operations"][0]["angle"] = {"parameter": "p", "offset": float("nan")}
        self.assertEqual(err(self.svc.create_circuit, base).code, "non_finite_number")
        base["operations"][0]["angle"] = {"parameter": "p", "extra": 1}
        e = err(self.svc.create_circuit, base)
        self.assertEqual((e.code, e.path), ("unknown_field", "operations[0].angle.extra"))
        base["operations"][0]["angle"] = {"coefficient": 1}
        e = err(self.svc.create_circuit, base)
        self.assertEqual((e.code, e.path), ("missing_field", "operations[0].angle.parameter"))

    def test_first_error_only_in_input_order(self):
        e = err(self.svc.create_circuit, {"qubit_count": 1, "operations": [
            {"gate": "x", "target": 0},
            {"gate": "nope", "target": 0},
            {"gate": "x", "target": 9},
        ]})
        self.assertEqual(e.path, "operations[1].gate")


class SimplifyTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def simplify_ops(self, ops, qubit_count=2, parameters=None):
        data = {"qubit_count": qubit_count, "operations": ops}
        if parameters is not None:
            data["parameters"] = parameters
        return self.svc.simplify(self.svc.create_circuit(data))

    def test_adjacent_involutions_cancel(self):
        for gate, op in (("x", {"gate": "x", "target": 0}),
                         ("h", {"gate": "h", "target": 1}),
                         ("cx", {"gate": "cx", "control": 0, "target": 1})):
            with self.subTest(gate=gate):
                out = self.simplify_ops([op, dict(op)])
                self.assertEqual(out["circuit"]["operations"], [])
                self.assertEqual(out["removed_operations"], 2)
                self.assertEqual(out["merged_operations"], 0)

    def test_strictly_adjacent_involutions_required(self):
        # 仅“相邻”的同位自逆门对消；中间隔着作用于其他位的门不发生交换对消。
        out = self.simplify_ops([
            {"gate": "x", "target": 0},
            {"gate": "x", "target": 1},
            {"gate": "x", "target": 0},
        ])
        self.assertEqual(len(out["circuit"]["operations"]), 3)
        self.assertEqual(out["removed_operations"], 0)

    def test_involution_blocked_by_interacting_op(self):
        out = self.simplify_ops([
            {"gate": "x", "target": 0},
            {"gate": "cx", "control": 1, "target": 0},
            {"gate": "x", "target": 0},
        ])
        self.assertEqual(len(out["circuit"]["operations"]), 3)
        self.assertEqual(out["removed_operations"], 0)

    def test_different_targets_do_not_cancel(self):
        out = self.simplify_ops([{"gate": "x", "target": 0}, {"gate": "x", "target": 1}])
        self.assertEqual(len(out["circuit"]["operations"]), 2)
        out = self.simplify_ops([
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "cx", "control": 1, "target": 0},
        ])
        self.assertEqual(len(out["circuit"]["operations"]), 2)

    def test_constant_rotations_merge_and_wrap(self):
        out = self.simplify_ops([
            {"gate": "rx", "target": 0, "angle": 1.0},
            {"gate": "rx", "target": 0, "angle": 2.0},
        ])
        self.assertEqual(out["circuit"]["operations"],
                         [{"gate": "rx", "target": 0, "angle": 3.0}])
        self.assertEqual(out["merged_operations"], 1)

    def test_rotation_merge_skips_noninteracting_ops(self):
        out = self.simplify_ops([
            {"gate": "rz", "target": 0, "angle": 1.0},
            {"gate": "x", "target": 1},
            {"gate": "rz", "target": 0, "angle": 0.5},
        ])
        self.assertEqual(out["circuit"]["operations"], [
            {"gate": "rz", "target": 0, "angle": 1.5},
            {"gate": "x", "target": 1},
        ])
        self.assertEqual(out["merged_operations"], 1)

    def test_rotation_blocked_by_interacting_op(self):
        out = self.simplify_ops([
            {"gate": "rz", "target": 0, "angle": 1.0},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rz", "target": 0, "angle": 0.5},
        ])
        self.assertEqual(len(out["circuit"]["operations"]), 3)

    def test_different_axes_and_targets_do_not_merge(self):
        out = self.simplify_ops([
            {"gate": "rx", "target": 0, "angle": 1.0},
            {"gate": "rz", "target": 0, "angle": 1.0},
        ])
        self.assertEqual(len(out["circuit"]["operations"]), 2)
        out = self.simplify_ops([
            {"gate": "rx", "target": 0, "angle": 1.0},
            {"gate": "rx", "target": 1, "angle": 1.0},
        ])
        self.assertEqual(len(out["circuit"]["operations"]), 2)

    def test_constants_wrapped_to_half_open_interval(self):
        out = self.simplify_ops([{"gate": "rx", "target": 0, "angle": 7.0}])
        angle = out["circuit"]["operations"][0]["angle"]
        self.assertTrue(-math.pi <= angle < math.pi)
        self.assertAlmostEqual(angle, 7.0 - 2 * math.pi, places=12)
        out = self.simplify_ops([{"gate": "rx", "target": 0, "angle": 2 * math.pi}])
        self.assertEqual(out["circuit"]["operations"], [])
        self.assertEqual(out["removed_operations"], 1)
        out = self.simplify_ops([{"gate": "rx", "target": 0, "angle": math.pi}])
        angle = out["circuit"]["operations"][0]["angle"]
        self.assertAlmostEqual(angle, -math.pi, places=12)
        self.assertTrue(-math.pi <= angle < math.pi)
        out = self.simplify_ops([{"gate": "rx", "target": 0, "angle": -math.pi}])
        self.assertAlmostEqual(out["circuit"]["operations"][0]["angle"], -math.pi, places=12)

    def test_constants_merge_to_zero_removed(self):
        out = self.simplify_ops([
            {"gate": "rx", "target": 0, "angle": 1.0},
            {"gate": "rx", "target": 0, "angle": -1.0},
        ])
        self.assertEqual(out["circuit"]["operations"], [])
        self.assertEqual(out["removed_operations"], 1)
        self.assertEqual(out["merged_operations"], 1)

    def test_parameterized_merge_same_parameter(self):
        out = self.simplify_ops([
            {"gate": "rz", "target": 0, "angle": {"parameter": "t", "coefficient": 1.0, "offset": 0.5}},
            {"gate": "rz", "target": 0, "angle": {"parameter": "t", "coefficient": 2.0, "offset": -0.25}},
        ], parameters=["t"])
        self.assertEqual(out["circuit"]["operations"], [
            {"gate": "rz", "target": 0,
             "angle": {"parameter": "t", "coefficient": 3.0, "offset": 0.25}},
        ])
        self.assertEqual(out["merged_operations"], 1)

    def test_parameterized_plus_constant(self):
        out = self.simplify_ops([
            {"gate": "rz", "target": 0, "angle": {"parameter": "t", "coefficient": 1.0, "offset": 0.5}},
            {"gate": "rz", "target": 0, "angle": 0.25},
        ], parameters=["t"])
        self.assertEqual(out["circuit"]["operations"][0]["angle"],
                         {"parameter": "t", "coefficient": 1.0, "offset": 0.75})

    def test_different_parameters_not_merged(self):
        out = self.simplify_ops([
            {"gate": "rz", "target": 0, "angle": {"parameter": "a", "coefficient": 1.0, "offset": 0.0}},
            {"gate": "rz", "target": 0, "angle": {"parameter": "b", "coefficient": 1.0, "offset": 0.0}},
        ], parameters=["a", "b"])
        self.assertEqual(len(out["circuit"]["operations"]), 2)
        self.assertEqual(out["merged_operations"], 0)

    def test_parameter_coefficients_cancel_to_wrapped_constant(self):
        out = self.simplify_ops([
            {"gate": "rz", "target": 0, "angle": {"parameter": "t", "coefficient": 1.0, "offset": 1.0}},
            {"gate": "rz", "target": 0, "angle": {"parameter": "t", "coefficient": -1.0, "offset": -1.0}},
        ], parameters=["t"])
        self.assertEqual(out["circuit"]["operations"], [])
        self.assertEqual(out["removed_operations"], 1)
        self.assertEqual(out["merged_operations"], 1)

    def test_idempotent(self):
        data = {"qubit_count": 3, "parameters": ["t"], "operations": [
            {"gate": "x", "target": 0}, {"gate": "x", "target": 0},
            {"gate": "rx", "target": 1, "angle": 9.0},
            {"gate": "rz", "target": 2, "angle": {"parameter": "t", "coefficient": 1.0, "offset": 0.0}},
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 1, "target": 2},
        ]}
        first = self.svc.simplify(data)
        second = self.svc.simplify(first["circuit"])
        self.assertEqual(second["circuit"], first["circuit"])
        self.assertEqual((second["removed_operations"], second["merged_operations"]), (0, 0))
        self.svc.create_circuit(first["circuit"])  # still valid canonical IR
        json.dumps(first["circuit"])

    def test_empty_circuit(self):
        out = self.svc.simplify({"qubit_count": 0})
        self.assertEqual(out["circuit"], {"qubit_count": 0, "parameters": [], "operations": []})
        self.assertEqual((out["removed_operations"], out["merged_operations"]), (0, 0))


class BindTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_bind_all_parameters(self):
        data = {"qubit_count": 1, "parameters": ["theta", "phi"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta", "coefficient": 2.0, "offset": 0.5}},
            {"gate": "rz", "target": 0, "angle": {"parameter": "phi"}},
            {"gate": "x", "target": 0},
        ]}
        snapshot = copy.deepcopy(data)
        result = self.svc.bind(data, {"theta": 1.0, "phi": -0.25})
        self.assertEqual(data, snapshot)
        self.assertEqual(result["parameters"], [])
        self.assertEqual(result["operations"][0]["angle"], 2.5)
        self.assertEqual(result["operations"][1]["angle"], -0.25)
        self.assertEqual(result["operations"][2], {"gate": "x", "target": 0})
        json.dumps(result)

    def test_bind_empty_circuit_with_empty_map(self):
        result = self.svc.bind({"qubit_count": 0}, {})
        self.assertEqual(result, {"qubit_count": 0, "parameters": [], "operations": []})

    def test_missing_parameter_sorted_first(self):
        data = {"qubit_count": 0, "parameters": ["zeta", "alpha"]}
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.bind(data, {})
        self.assertEqual(ctx.exception.code, "missing_parameter")
        self.assertEqual(ctx.exception.path, "alpha")

    def test_unknown_parameter_sorted_first(self):
        data = {"qubit_count": 0}
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.bind(data, {"zeta": 1.0, "alpha": 2.0})
        self.assertEqual(ctx.exception.code, "unknown_parameter")
        self.assertEqual(ctx.exception.path, "alpha")

    def test_exact_match_required(self):
        data = {"qubit_count": 0, "parameters": ["a", "b"]}
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.bind(data, {"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertEqual(ctx.exception.code, "unknown_parameter")

    def test_invalid_binding_values(self):
        data = {"qubit_count": 0, "parameters": ["a"]}
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.bind(data, {"a": True})
        self.assertEqual(ctx.exception.code, "invalid_type")
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.bind(data, {"a": float("nan")})
        self.assertEqual(ctx.exception.code, "non_finite_number")
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.bind(data, [])
        self.assertEqual(ctx.exception.code, "invalid_type")

    def test_bind_does_not_require_simplification_and_keeps_angles(self):
        data = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
            {"gate": "rx", "target": 0, "angle": 0.5},
        ]}
        result = self.svc.bind(data, {"t": 1.0})
        self.assertEqual([op["angle"] for op in result["operations"]], [1.0, 0.5])


class HealthCompatTest(unittest.TestCase):
    def test_health_unchanged(self):
        payload = Service().health()
        self.assertEqual(payload, {"status": "ok", "service": "qubitfabric", "version": "0.1.0"})
        json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
