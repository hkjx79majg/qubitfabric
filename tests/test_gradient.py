import copy
import json
import math
import unittest

from qubitfabric.service import (
    CircuitValidationError,
    ParameterBindingError,
    Service,
    SimulationError,
)


def rx_circuit(angle, extra_parameters=()):
    return {
        "qubit_count": 1,
        "parameters": ["theta", *extra_parameters],
        "operations": [{"gate": "rx", "target": 0, "angle": angle}],
    }


class GradientValueTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def gradients(self, circuit, observables, values=None):
        result = self.svc.gradient(circuit, observables, values=values)
        return result

    def test_rx_z_gradient_matches_minus_sin(self):
        circuit = rx_circuit({"parameter": "theta"})
        result = self.gradients(circuit, ["Z"], values={"theta": 0.5})
        self.assertEqual(result["qubit_count"], 1)
        self.assertEqual(result["parameters"], ["theta"])
        item = result["results"][0]
        self.assertEqual(item["observable"], "Z")
        self.assertAlmostEqual(item["expectation"], math.cos(0.5), places=12)
        self.assertAlmostEqual(item["gradients"]["theta"], -math.sin(0.5), places=12)

    def test_coefficient_and_offset_scale_gradient(self):
        circuit = rx_circuit({"parameter": "theta", "coefficient": 2.0, "offset": 0.3})
        item = self.gradients(circuit, ["Z"], values={"theta": 0.7})["results"][0]
        angle = 2.0 * 0.7 + 0.3
        self.assertAlmostEqual(item["expectation"], math.cos(angle), places=12)
        self.assertAlmostEqual(item["gradients"]["theta"], -2.0 * math.sin(angle), places=12)

    def test_negative_coefficient(self):
        circuit = rx_circuit({"parameter": "theta", "coefficient": -1.5})
        item = self.gradients(circuit, ["Z"], values={"theta": 0.4})["results"][0]
        self.assertAlmostEqual(item["gradients"]["theta"], 1.5 * math.sin(-0.6), places=12)

    def test_rz_gradient_after_hadamard(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [
                {"gate": "h", "target": 0},
                {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}},
            ],
        }
        item = self.gradients(circuit, ["X", "Y"], values={"theta": 0.7})["results"]
        self.assertAlmostEqual(item[0]["gradients"]["theta"], -math.sin(0.7), places=12)
        self.assertAlmostEqual(item[1]["gradients"]["theta"], math.cos(0.7), places=12)

    def test_same_parameter_in_multiple_gates_accumulates(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [
                {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
                {"gate": "rx", "target": 0, "angle": {"parameter": "theta", "coefficient": 2.0}},
            ],
        }
        item = self.gradients(circuit, ["Z"], values={"theta": 0.3})["results"][0]
        self.assertAlmostEqual(item["expectation"], math.cos(0.9), places=12)
        self.assertAlmostEqual(item["gradients"]["theta"], -3.0 * math.sin(0.9), places=12)

    def test_unused_parameter_has_zero_gradient(self):
        circuit = rx_circuit({"parameter": "theta"}, extra_parameters=["unused"])
        result = self.gradients(circuit, ["Z"], values={"theta": 0.5, "unused": 9.0})
        self.assertEqual(result["parameters"], ["theta", "unused"])
        gradients = result["results"][0]["gradients"]
        self.assertEqual(list(gradients), ["theta", "unused"])
        self.assertEqual(gradients["unused"], 0.0)

    def test_zero_coefficient_expression_counts_as_unused(self):
        circuit = rx_circuit({"parameter": "theta", "coefficient": 0.0, "offset": 0.5})
        item = self.gradients(circuit, ["Z"], values={"theta": 3.0})["results"][0]
        self.assertAlmostEqual(item["expectation"], math.cos(0.5), places=12)
        self.assertEqual(item["gradients"], {"theta": 0.0})

    def test_parameter_order_and_observable_order_preserved(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["b", "a"],
            "operations": [
                {"gate": "rx", "target": 0, "angle": {"parameter": "a"}},
                {"gate": "rz", "target": 0, "angle": {"parameter": "b"}},
            ],
        }
        result = self.gradients(circuit, ["Z", "X", "Z"], values={"a": 0.2, "b": 0.9})
        self.assertEqual(result["parameters"], ["b", "a"])
        self.assertEqual([r["observable"] for r in result["results"]], ["Z", "X", "Z"])
        for item in result["results"]:
            self.assertEqual(list(item["gradients"]), ["b", "a"])
        self.assertEqual(result["results"][0], result["results"][2])

    def test_parameterless_circuit_allows_omitted_or_empty_values(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        omitted = self.gradients(circuit, ["Z"])
        empty = self.gradients(circuit, ["Z"], values={})
        self.assertEqual(omitted, empty)
        self.assertEqual(omitted["parameters"], [])
        self.assertEqual(omitted["results"], [{"observable": "Z", "expectation": -1.0, "gradients": {}}])

    def test_zero_qubit_circuit(self):
        result = self.gradients({"qubit_count": 0}, [""])
        self.assertEqual(result["results"], [{"observable": "", "expectation": 1.0, "gradients": {}}])

    def test_negative_zero_is_normalized(self):
        # 对称点上差分为精确 0.0，负系数会产生 -0.0，必须规范为 0.0。
        circuit = rx_circuit({"parameter": "theta", "coefficient": -1.0})
        gradients = self.gradients(circuit, ["Z"], values={"theta": 0.0})["results"][0]["gradients"]
        self.assertEqual(gradients["theta"], 0.0)
        self.assertEqual(json.dumps(gradients), '{"theta": 0.0}')

    def test_expectation_matches_estimate_expectation(self):
        circuit = {
            "qubit_count": 2,
            "parameters": ["theta"],
            "operations": [
                {"gate": "h", "target": 0},
                {"gate": "cx", "control": 0, "target": 1},
                {"gate": "rz", "target": 1, "angle": {"parameter": "theta", "offset": 0.1}},
            ],
        }
        observables = ["ZZ", "XX", "YY", "ZI"]
        gradient = self.gradients(circuit, observables, values={"theta": 0.4})
        exact = self.svc.expectation(circuit, observables, values={"theta": 0.4})
        for g_item, e_item in zip(gradient["results"], exact["results"]):
            self.assertEqual(g_item["observable"], e_item["observable"])
            self.assertEqual(g_item["expectation"], e_item["expectation"])

    def test_input_not_mutated_and_result_json_serializable(self):
        circuit = rx_circuit({"parameter": "theta", "coefficient": 2.0, "offset": 0.1})
        observables = ["Z", "X"]
        values = {"theta": 0.3}
        snapshot = (copy.deepcopy(circuit), list(observables), dict(values))
        result = self.gradients(circuit, observables, values=values)
        self.assertEqual((circuit, observables, values), snapshot)
        json.dumps(result, sort_keys=True)


class GradientValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_invalid_circuit_raises_circuit_validation_error(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.gradient({"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]}, ["Z"])

    def test_circuit_validated_before_observables(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.gradient({"qubit_count": -1}, [])

    def test_binding_errors_match_expectation(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [{"gate": "rz", "target": 0, "angle": {"parameter": "theta"}}],
        }
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.gradient(circuit, ["Z"])
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("missing_parameter", "theta"))

        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.gradient({"qubit_count": 1}, ["Z"], values={"nope": 1})
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("unknown_parameter", "nope"))

        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.gradient(circuit, ["Z"], values={"theta": "x"})
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("invalid_type", "theta"))

        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.gradient(circuit, ["Z"], values={"theta": float("nan")})
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("non_finite_number", "theta"))

        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.gradient(circuit, ["Z"], values=1)
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("invalid_type", "$"))

    def test_binding_validated_before_observables(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [{"gate": "rz", "target": 0, "angle": {"parameter": "theta"}}],
        }
        with self.assertRaises(ParameterBindingError):
            self.svc.gradient(circuit, [], values={})

    def test_invalid_observables(self):
        with self.assertRaises(SimulationError) as ctx:
            self.svc.gradient({"qubit_count": 1}, [])
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("invalid_observables", "observables"))

        with self.assertRaises(SimulationError) as ctx:
            self.svc.gradient({"qubit_count": 1}, ["ZZ"])
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("invalid_observable", "observables[0]"))

    def test_state_space_too_large(self):
        with self.assertRaises(SimulationError) as ctx:
            self.svc.gradient({"qubit_count": 21}, ["I" * 21])
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("state_space_too_large", "qubit_count"))
        # 边界：20 个量子位仍然可用。
        result = self.svc.gradient({"qubit_count": 20}, ["I" * 20])
        self.assertEqual(result["results"][0]["expectation"], 1.0)


if __name__ == "__main__":
    unittest.main()
