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


def bind_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ParameterBindingError as exc:
        return exc
    raise AssertionError("ParameterBindingError not raised")


def sim_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except SimulationError as exc:
        return exc
    raise AssertionError("SimulationError not raised")


class ExactGradientTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def gradients(self, circuit, observables, values=None):
        result = self.svc.gradient(circuit, observables, values=values)
        return result

    def test_rx_z_gradient_matches_analytic(self):
        theta = 0.3
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        result = self.gradients(circuit, ["Z"], values={"theta": theta})
        item = result["results"][0]
        self.assertAlmostEqual(item["expectation"], math.cos(theta), places=12)
        self.assertAlmostEqual(item["gradients"]["theta"], -math.sin(theta), places=12)

    def test_rz_xy_gradients_match_analytic(self):
        theta = 0.7
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "h", "target": 0},
            {"gate": "rz", "target": 0, "angle": {"parameter": "t"}},
        ]}
        result = self.gradients(circuit, ["X", "Y"], values={"t": theta})
        x_item, y_item = result["results"]
        self.assertAlmostEqual(x_item["expectation"], math.cos(theta), places=12)
        self.assertAlmostEqual(x_item["gradients"]["t"], -math.sin(theta), places=12)
        self.assertAlmostEqual(y_item["expectation"], math.sin(theta), places=12)
        self.assertAlmostEqual(y_item["gradients"]["t"], math.cos(theta), places=12)

    def test_coefficient_and_offset_scale_the_shift_rule(self):
        # 角度 2θ+0.1：Z 期望 cos(2θ+0.1)，导数 -2 sin(2θ+0.1)。
        theta = 0.4
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "theta", "coefficient": 2.0, "offset": 0.1}},
        ]}
        item = self.gradients(circuit, ["Z"], values={"theta": theta})["results"][0]
        angle = 2.0 * theta + 0.1
        self.assertAlmostEqual(item["expectation"], math.cos(angle), places=12)
        self.assertAlmostEqual(item["gradients"]["theta"], -2.0 * math.sin(angle), places=12)

    def test_negative_coefficient(self):
        theta = 0.9
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "t", "coefficient": -1.5}},
        ]}
        item = self.gradients(circuit, ["Z"], values={"t": theta})["results"][0]
        angle = -1.5 * theta
        self.assertAlmostEqual(item["expectation"], math.cos(angle), places=12)
        self.assertAlmostEqual(item["gradients"]["t"], 1.5 * math.sin(angle), places=12)

    def test_shared_parameter_accumulates_gate_contributions(self):
        # 两个 rx 共用同一参数：总角 2θ，Z 期望 cos(2θ)，导数 -2 sin(2θ)。
        theta = 0.6
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        item = self.gradients(circuit, ["Z"], values={"t": theta})["results"][0]
        self.assertAlmostEqual(item["expectation"], math.cos(2.0 * theta), places=12)
        self.assertAlmostEqual(item["gradients"]["t"], -2.0 * math.sin(2.0 * theta), places=12)

    def test_unused_parameter_has_zero_gradient(self):
        circuit = {"qubit_count": 1, "parameters": ["used", "unused"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "used"}},
        ]}
        result = self.gradients(circuit, ["Z"], values={"used": 0.5, "unused": 1.25})
        self.assertEqual(result["parameters"], ["used", "unused"])
        gradients = result["results"][0]["gradients"]
        self.assertEqual(list(gradients), ["used", "unused"])
        self.assertEqual(gradients["unused"], 0.0)
        self.assertAlmostEqual(gradients["used"], -math.sin(0.5), places=12)

    def test_constant_angles_and_other_gates_contribute_nothing(self):
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "x", "target": 0},
            {"gate": "rx", "target": 0, "angle": 0.25},
            {"gate": "h", "target": 1},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rz", "target": 1, "angle": {"parameter": "t"}},
        ]}
        result = self.gradients(circuit, ["ZZ", "XI"], values={"t": 0.8})
        # 与 svc.expectation 的精确值一致。
        exact = self.svc.expectation(circuit, ["ZZ", "XI"], values={"t": 0.8})
        for got, want in zip(result["results"], exact["results"]):
            self.assertEqual(got["expectation"], want["expectation"])

    def test_results_keep_observable_order_and_duplicates(self):
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        result = self.gradients(circuit, ["Z", "X", "Z"], values={"t": 0.3})
        self.assertEqual([r["observable"] for r in result["results"]], ["Z", "X", "Z"])
        self.assertEqual(result["results"][0], result["results"][2])
        self.assertEqual(result["qubit_count"], 1)

    def test_no_parameter_circuit_allows_omitted_or_empty_values(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        omitted = self.svc.gradient(circuit, ["X"])
        empty = self.svc.gradient(circuit, ["X"], values={})
        self.assertEqual(omitted, empty)
        self.assertEqual(omitted["parameters"], [])
        self.assertEqual(omitted["results"][0]["gradients"], {})
        self.assertAlmostEqual(omitted["results"][0]["expectation"], 1.0, places=12)

    def test_zero_qubit_circuit(self):
        result = self.svc.gradient({"qubit_count": 0}, [""])
        self.assertEqual(result["results"], [
            {"observable": "", "expectation": 1.0, "gradients": {}},
        ])

    def test_gradient_matches_central_finite_difference(self):
        circuit = {"qubit_count": 2, "parameters": ["a", "b"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "a", "coefficient": 2.0, "offset": 0.3}},
            {"gate": "h", "target": 1},
            {"gate": "rz", "target": 1,
             "angle": {"parameter": "b", "coefficient": -0.5}},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rx", "target": 1, "angle": {"parameter": "a"}},
            {"gate": "rz", "target": 0, "angle": 0.7},
        ]}
        observables = ["ZZ", "XI", "IY", "YX"]
        values = {"a": 0.4, "b": 1.1}
        result = self.svc.gradient(circuit, observables, values=values)

        eps = 1e-6
        for k, item in enumerate(result["results"]):
            for name in ("a", "b"):
                plus = dict(values, **{name: values[name] + eps})
                minus = dict(values, **{name: values[name] - eps})
                e_plus = self.svc.expectation(circuit, observables, values=plus)["results"][k]["expectation"]
                e_minus = self.svc.expectation(circuit, observables, values=minus)["results"][k]["expectation"]
                numeric = (e_plus - e_minus) / (2.0 * eps)
                self.assertAlmostEqual(item["gradients"][name], numeric, places=5)

    def test_input_not_mutated_and_result_json_serializable(self):
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        observables = ["Z", "X"]
        values = {"t": 0.5}
        snapshot = (copy.deepcopy(circuit), list(observables), dict(values))
        result = self.svc.gradient(circuit, observables, values=values)
        self.assertEqual((circuit, observables, values), snapshot)
        payload = json.dumps(result, sort_keys=True)
        self.assertNotIn("-0.0", payload)


class GradientValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}},
        ]}

    def test_circuit_validation_is_reused(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.gradient({"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]}, ["Z"])

    def test_missing_and_unknown_parameter_raise_binding_error(self):
        e = bind_err(self.svc.gradient, self.circuit, ["Z"])
        self.assertEqual((e.code, e.path), ("missing_parameter", "theta"))
        e = bind_err(self.svc.gradient, {"qubit_count": 1}, ["Z"], values={"nope": 1})
        self.assertEqual((e.code, e.path), ("unknown_parameter", "nope"))

    def test_invalid_values_type_and_non_finite(self):
        e = bind_err(self.svc.gradient, self.circuit, ["Z"], values=[("theta", 1.0)])
        self.assertEqual((e.code, e.path), ("invalid_type", "$"))
        e = bind_err(self.svc.gradient, self.circuit, ["Z"], values={"theta": "x"})
        self.assertEqual((e.code, e.path), ("invalid_type", "theta"))
        e = bind_err(self.svc.gradient, self.circuit, ["Z"], values={"theta": math.inf})
        self.assertEqual((e.code, e.path), ("non_finite_number", "theta"))

    def test_observables_must_be_non_empty_array(self):
        for bad in (None, "Z", {"0": "Z"}, [], 1):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.gradient, {"qubit_count": 1}, bad)
                self.assertEqual((e.code, e.path), ("invalid_observables", "observables"))

    def test_invalid_observable_items(self):
        for observables, path in [(["Z", 1], "observables[1]"), (["ZZ"], "observables[0]"), (["z"], "observables[0]")]:
            with self.subTest(observables=observables):
                e = sim_err(self.svc.gradient, {"qubit_count": 1}, observables)
                self.assertEqual((e.code, e.path), ("invalid_observable", path))

    def test_state_space_too_large(self):
        e = sim_err(self.svc.gradient, {"qubit_count": 21}, ["I" * 21])
        self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))
        result = self.svc.gradient({"qubit_count": 20}, ["I" * 20])
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_validation_order_matches_expectation(self):
        # 电路错误先于绑定错误，绑定错误先于 observables 错误。
        bad_circuit = {"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]}
        with self.assertRaises(CircuitValidationError):
            self.svc.gradient(bad_circuit, [], values={"theta": 1})
        e = bind_err(self.svc.gradient, self.circuit, [])
        self.assertEqual(e.code, "missing_parameter")


if __name__ == "__main__":
    unittest.main()
