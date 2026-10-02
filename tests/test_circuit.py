import json
import math
import unittest

from qubitfabric import CircuitValidationError, ParameterBindingError
from qubitfabric.service import Service


def normalize(spec):
    return Service().normalize(spec)


def simplify(circuit):
    return Service().simplify(circuit)


def bind(circuit, bindings):
    return Service().bind(circuit, bindings)


class NormalizeTest(unittest.TestCase):
    def test_defaults_are_filled(self):
        result = normalize({"qubit_count": 2})
        self.assertEqual(result, {"qubit_count": 2, "parameters": [], "operations": []})

    def test_zero_qubit_empty_circuit(self):
        result = normalize({"qubit_count": 0, "parameters": [], "operations": []})
        self.assertEqual(result, {"qubit_count": 0, "parameters": [], "operations": []})

    def test_all_gates_normalize(self):
        spec = {
            "qubit_count": 2,
            "parameters": ["theta"],
            "operations": [
                {"gate": "x", "target": 0},
                {"gate": "h", "target": 1},
                {"gate": "rx", "target": 0, "angle": 1},
                {"gate": "rz", "target": 1, "angle": "theta"},
                {"gate": "cx", "control": 0, "target": 1},
            ],
        }
        result = normalize(spec)
        self.assertEqual(
            result["operations"],
            [
                {"gate": "x", "target": 0},
                {"gate": "h", "target": 1},
                {"gate": "rx", "target": 0, "angle": 1.0},
                {"gate": "rz", "target": 1,
                 "angle": {"parameter": "theta", "coefficient": 1.0, "offset": 0.0}},
                {"gate": "cx", "control": 0, "target": 1},
            ],
        )

    def test_semantically_equal_inputs_produce_identical_forms(self):
        base = normalize({
            "qubit_count": 1,
            "parameters": ["t"],
            "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "t"}}],
        })
        variants = [
            {"qubit_count": 1, "parameters": ["t"],
             "operations": [{"gate": "rx", "target": 0, "angle": "t"}]},
            {"qubit_count": 1, "parameters": ["t"],
             "operations": [{"gate": "rx", "target": 0,
                             "angle": {"parameter": "t", "coefficient": 1, "offset": 0}}]},
            {"qubit_count": 1, "parameters": ["t"],
             "operations": [{"gate": "rx", "target": 0,
                             "angle": {"parameter": "t", "coefficient": 1.0, "offset": 0.0}}]},
        ]
        for variant in variants:
            self.assertEqual(normalize(variant), base)
        self.assertEqual(
            normalize({"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": 1}]}),
            normalize({"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": 1.0}]}),
        )

    def test_input_is_not_modified(self):
        spec = {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": 1}]}
        snapshot = json.loads(json.dumps(spec))
        normalize(spec)
        simplify(spec)
        bind({"qubit_count": 1, "parameters": ["t"],
              "operations": [{"gate": "rz", "target": 0, "angle": "t"}]}, {"t": 0.5})
        self.assertEqual(spec, snapshot)

    def test_result_is_json_native(self):
        result = normalize({
            "qubit_count": 2,
            "parameters": ["t"],
            "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "t", "offset": 0.25}}],
        })
        self.assertEqual(json.loads(json.dumps(result)), result)


class ValidationErrorTest(unittest.TestCase):
    def assert_error(self, spec, code, path):
        with self.assertRaises(CircuitValidationError) as ctx:
            normalize(spec)
        self.assertEqual(ctx.exception.code, code, ctx.exception)
        self.assertEqual(ctx.exception.path, path, ctx.exception)

    def test_top_level_type(self):
        self.assert_error([], "invalid_type", [])
        self.assert_error(None, "invalid_type", [])
        self.assert_error("circuit", "invalid_type", [])

    def test_unknown_and_missing_top_level_fields(self):
        self.assert_error({"qubit_count": 1, "extra": 1}, "unknown_field", ["extra"])
        self.assert_error({}, "missing_field", ["qubit_count"])

    def test_qubit_count_validation(self):
        self.assert_error({"qubit_count": -1}, "invalid_value", ["qubit_count"])
        self.assert_error({"qubit_count": 1.5}, "invalid_type", ["qubit_count"])
        self.assert_error({"qubit_count": True}, "invalid_type", ["qubit_count"])

    def test_parameter_declaration_errors(self):
        self.assert_error({"qubit_count": 1, "parameters": "t"}, "invalid_type", ["parameters"])
        self.assert_error({"qubit_count": 1, "parameters": [1]}, "invalid_type", ["parameters", 0])
        self.assert_error({"qubit_count": 1, "parameters": [""]}, "invalid_parameter_name", ["parameters", 0])
        self.assert_error(
            {"qubit_count": 1, "parameters": ["a", "a"]},
            "duplicate_parameter", ["parameters", 1],
        )

    def test_gate_errors(self):
        self.assert_error({"qubit_count": 1, "operations": [[]]}, "invalid_type", ["operations", 0])
        self.assert_error(
            {"qubit_count": 1, "operations": [{}]}, "missing_field", ["operations", 0, "gate"]
        )
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": 1}]}, "invalid_type", ["operations", 0, "gate"]
        )
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "toffoli"}]},
            "unknown_gate", ["operations", 0, "gate"],
        )

    def test_operation_field_mismatch(self):
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "x"}]},
            "missing_field", ["operations", 0, "target"],
        )
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "x", "target": 0, "angle": 1}]},
            "unknown_field", ["operations", 0, "angle"],
        )
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0}]},
            "missing_field", ["operations", 0, "angle"],
        )
        self.assert_error(
            {"qubit_count": 2, "operations": [{"gate": "cx", "control": 0}]},
            "missing_field", ["operations", 0, "target"],
        )

    def test_qubit_out_of_range(self):
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "x", "target": 1}]},
            "qubit_out_of_range", ["operations", 0, "target"],
        )
        self.assert_error(
            {"qubit_count": 2, "operations": [{"gate": "cx", "control": 0, "target": 2}]},
            "qubit_out_of_range", ["operations", 0, "target"],
        )

    def test_cx_same_qubits(self):
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "cx", "control": 0, "target": 0}]},
            "same_qubits", ["operations", 0, "target"],
        )

    def test_bool_masquerading_as_number(self):
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "x", "target": True}]},
            "invalid_type", ["operations", 0, "target"],
        )
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": False}]},
            "invalid_type", ["operations", 0, "angle"],
        )

    def test_non_finite_numbers(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assert_error(
                {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": bad}]},
                "non_finite_number", ["operations", 0, "angle"],
            )
        self.assert_error(
            {"qubit_count": 1, "parameters": ["t"],
             "operations": [{"gate": "rx", "target": 0,
                             "angle": {"parameter": "t", "coefficient": float("inf")}}]},
            "non_finite_number", ["operations", 0, "angle", "coefficient"],
        )

    def test_undefined_parameter(self):
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "rz", "target": 0, "angle": "t"}]},
            "undefined_parameter", ["operations", 0, "angle"],
        )
        self.assert_error(
            {"qubit_count": 1, "operations": [{"gate": "rz", "target": 0, "angle": {"parameter": "t"}}]},
            "undefined_parameter", ["operations", 0, "angle", "parameter"],
        )

    def test_only_first_error_in_input_order_is_reported(self):
        spec = {"qubit_count": 1, "operations": [
            {"gate": "x", "target": 5},
            {"gate": "nope"},
        ]}
        self.assert_error(spec, "qubit_out_of_range", ["operations", 0, "target"])


class SimplifyTest(unittest.TestCase):
    def ops(self, circuit):
        return circuit["operations"]

    def test_cancels_adjacent_self_inverse_pairs(self):
        result = simplify({"qubit_count": 2, "operations": [
            {"gate": "x", "target": 0},
            {"gate": "x", "target": 0},
            {"gate": "h", "target": 1},
            {"gate": "h", "target": 1},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "cx", "control": 0, "target": 1},
        ]})
        self.assertEqual(self.ops(result["circuit"]), [])
        self.assertEqual(result["removed_operations"], 6)
        self.assertEqual(result["merged_operations"], 0)

    def test_cancellation_requires_same_qubits_and_adjacency(self):
        result = simplify({"qubit_count": 2, "operations": [
            {"gate": "x", "target": 0},
            {"gate": "x", "target": 1},
            {"gate": "h", "target": 0},
            {"gate": "x", "target": 0},
            {"gate": "h", "target": 0},
        ]})
        # x on different qubits and h separated by x: nothing cancels.
        self.assertEqual(self.ops(result["circuit"]), [
            {"gate": "x", "target": 0},
            {"gate": "x", "target": 1},
            {"gate": "h", "target": 0},
            {"gate": "x", "target": 0},
            {"gate": "h", "target": 0},
        ])
        self.assertEqual(result["removed_operations"], 0)

    def test_cx_cancellation_is_direction_sensitive(self):
        result = simplify({"qubit_count": 2, "operations": [
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "cx", "control": 1, "target": 0},
        ]})
        self.assertEqual(len(self.ops(result["circuit"])), 2)
        self.assertEqual(result["removed_operations"], 0)

    def test_merges_constant_rotations(self):
        result = simplify({"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 0.5},
            {"gate": "rx", "target": 0, "angle": 0.25},
        ]})
        self.assertEqual(self.ops(result["circuit"]),
                         [{"gate": "rx", "target": 0, "angle": 0.75}])
        self.assertEqual(result["merged_operations"], 1)
        self.assertEqual(result["removed_operations"], 0)

    def test_merge_spans_unrelated_qubits_only(self):
        result = simplify({"qubit_count": 2, "operations": [
            {"gate": "rz", "target": 0, "angle": 0.5},
            {"gate": "h", "target": 1},
            {"gate": "rz", "target": 0, "angle": 0.5},
        ]})
        self.assertEqual(self.ops(result["circuit"]), [
            {"gate": "rz", "target": 0, "angle": 1.0},
            {"gate": "h", "target": 1},
        ])
        # An intervening operation on the same qubit blocks the merge.
        blocked = simplify({"qubit_count": 1, "operations": [
            {"gate": "rz", "target": 0, "angle": 0.5},
            {"gate": "h", "target": 0},
            {"gate": "rz", "target": 0, "angle": 0.5},
        ]})
        self.assertEqual(len(self.ops(blocked["circuit"])), 3)
        self.assertEqual(blocked["merged_operations"], 0)

    def test_different_axes_do_not_merge(self):
        result = simplify({"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 0.5},
            {"gate": "rz", "target": 0, "angle": 0.5},
        ]})
        self.assertEqual(len(self.ops(result["circuit"])), 2)
        self.assertEqual(result["merged_operations"], 0)

    def test_constant_angles_reduce_to_half_open_interval(self):
        result = simplify({"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 3 * math.pi},
        ]})
        self.assertEqual(self.ops(result["circuit"]),
                         [{"gate": "rx", "target": 0, "angle": -math.pi}])
        exact = simplify({"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": math.pi},
        ]})
        self.assertEqual(self.ops(exact["circuit"]),
                         [{"gate": "rx", "target": 0, "angle": -math.pi}])

    def test_zero_angle_rotations_are_deleted(self):
        result = simplify({"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 0.0},
            {"gate": "rz", "target": 0, "angle": 2 * math.pi},
        ]})
        self.assertEqual(self.ops(result["circuit"]), [])
        self.assertEqual(result["removed_operations"], 2)

    def test_merged_rotation_that_cancels_is_deleted(self):
        result = simplify({"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 0.5},
            {"gate": "rx", "target": 0, "angle": -0.5},
        ]})
        self.assertEqual(self.ops(result["circuit"]), [])
        self.assertEqual(result["merged_operations"], 1)
        self.assertEqual(result["removed_operations"], 1)

    def test_parameterized_rotations_merge_exactly(self):
        result = simplify({"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": "t"},
            {"gate": "rx", "target": 0, "angle": {"parameter": "t", "coefficient": 2, "offset": 0.5}},
        ]})
        self.assertEqual(self.ops(result["circuit"]), [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "t", "coefficient": 3.0, "offset": 0.5}},
        ])
        self.assertEqual(result["merged_operations"], 1)

    def test_constant_merges_into_parameterized(self):
        result = simplify({"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rz", "target": 0, "angle": 0.25},
            {"gate": "rz", "target": 0, "angle": "t"},
        ]})
        self.assertEqual(self.ops(result["circuit"]), [
            {"gate": "rz", "target": 0,
             "angle": {"parameter": "t", "coefficient": 1.0, "offset": 0.25}},
        ])

    def test_different_parameters_keep_original_order(self):
        result = simplify({"qubit_count": 1, "parameters": ["a", "b"], "operations": [
            {"gate": "rx", "target": 0, "angle": "a"},
            {"gate": "rx", "target": 0, "angle": "b"},
        ]})
        self.assertEqual(self.ops(result["circuit"]), [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "a", "coefficient": 1.0, "offset": 0.0}},
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "b", "coefficient": 1.0, "offset": 0.0}},
        ])
        self.assertEqual(result["merged_operations"], 0)

    def test_simplify_is_idempotent(self):
        spec = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "x", "target": 0},
            {"gate": "rx", "target": 0, "angle": 0.0},
            {"gate": "x", "target": 0},
            {"gate": "rx", "target": 1, "angle": 0.5},
            {"gate": "h", "target": 0},
            {"gate": "rx", "target": 1, "angle": 0.25},
            {"gate": "rz", "target": 1, "angle": "t"},
            {"gate": "rz", "target": 1, "angle": "t"},
        ]}
        once = simplify(spec)
        twice = simplify(once["circuit"])
        self.assertEqual(once["circuit"], twice["circuit"])
        self.assertEqual(twice["removed_operations"], 0)
        self.assertEqual(twice["merged_operations"], 0)
        # The x pair becomes adjacent once the zero-angle rx is deleted;
        # merged rotations keep the position of the first rotation.
        self.assertEqual(once["circuit"]["operations"], [
            {"gate": "rx", "target": 1, "angle": 0.75},
            {"gate": "h", "target": 0},
            {"gate": "rz", "target": 1,
             "angle": {"parameter": "t", "coefficient": 2.0, "offset": 0.0}},
        ])

    def test_result_is_valid_normalized_ir(self):
        spec = {"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 0.5},
            {"gate": "rx", "target": 0, "angle": 0.5},
        ]}
        result = simplify(spec)
        self.assertEqual(normalize(result["circuit"]), result["circuit"])
        json.dumps(result, sort_keys=True)

    def test_zero_qubit_empty_circuit_simplifies(self):
        result = simplify({"qubit_count": 0})
        self.assertEqual(result["circuit"],
                         {"qubit_count": 0, "parameters": [], "operations": []})
        self.assertEqual(result["removed_operations"], 0)
        self.assertEqual(result["merged_operations"], 0)


class BindTest(unittest.TestCase):
    def test_bind_replaces_all_parameters(self):
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t", "coefficient": 2, "offset": 0.5}},
            {"gate": "rz", "target": 1, "angle": "t"},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        result = bind(circuit, {"t": 0.25})
        self.assertEqual(result, {"qubit_count": 2, "parameters": [], "operations": [
            {"gate": "rx", "target": 0, "angle": 1.0},
            {"gate": "rz", "target": 1, "angle": 0.25},
            {"gate": "cx", "control": 0, "target": 1},
        ]})
        json.dumps(result, sort_keys=True)

    def test_bind_does_not_modify_input(self):
        circuit = {"qubit_count": 1, "parameters": ["t"],
                   "operations": [{"gate": "rx", "target": 0, "angle": "t"}]}
        snapshot = json.loads(json.dumps(circuit))
        bind(circuit, {"t": 1.0})
        self.assertEqual(circuit, snapshot)

    def test_bind_result_has_no_parameter_declarations(self):
        result = bind({"qubit_count": 0, "parameters": ["t"]}, {"t": 3})
        self.assertEqual(result["parameters"], [])

    def test_missing_parameter(self):
        with self.assertRaises(ParameterBindingError) as ctx:
            bind({"qubit_count": 1, "parameters": ["t"],
                  "operations": [{"gate": "rx", "target": 0, "angle": "t"}]}, {})
        self.assertEqual(ctx.exception.code, "missing_parameter")
        self.assertEqual(ctx.exception.parameter, "t")

    def test_unknown_parameter(self):
        with self.assertRaises(ParameterBindingError) as ctx:
            bind({"qubit_count": 0}, {"t": 1.0})
        self.assertEqual(ctx.exception.code, "unknown_parameter")
        self.assertEqual(ctx.exception.parameter, "t")

    def test_multiple_names_choose_lexicographically_first(self):
        with self.assertRaises(ParameterBindingError) as ctx:
            bind({"qubit_count": 0, "parameters": ["b", "a"]}, {})
        self.assertEqual(ctx.exception.code, "missing_parameter")
        self.assertEqual(ctx.exception.parameter, "a")
        with self.assertRaises(ParameterBindingError) as ctx:
            bind({"qubit_count": 0}, {"b": 1, "a": 2})
        self.assertEqual(ctx.exception.code, "unknown_parameter")
        self.assertEqual(ctx.exception.parameter, "a")

    def test_binding_values_must_be_finite_numbers(self):
        with self.assertRaises(CircuitValidationError) as ctx:
            bind({"qubit_count": 0, "parameters": ["t"]}, {"t": float("nan")})
        self.assertEqual(ctx.exception.code, "non_finite_number")
        with self.assertRaises(CircuitValidationError) as ctx:
            bind({"qubit_count": 0, "parameters": ["t"]}, {"t": True})
        self.assertEqual(ctx.exception.code, "invalid_type")

    def test_zero_qubit_empty_circuit_binds(self):
        result = bind({"qubit_count": 0}, {})
        self.assertEqual(result, {"qubit_count": 0, "parameters": [], "operations": []})


class ServiceSurfaceTest(unittest.TestCase):
    def test_health_unchanged(self):
        payload = Service().health()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "qubitfabric")

    def test_errors_are_publicly_importable(self):
        import qubitfabric
        import qubitfabric.service
        self.assertIs(qubitfabric.CircuitValidationError, CircuitValidationError)
        self.assertIs(qubitfabric.service.CircuitValidationError, CircuitValidationError)
        self.assertIs(qubitfabric.ParameterBindingError, ParameterBindingError)


if __name__ == "__main__":
    unittest.main()
