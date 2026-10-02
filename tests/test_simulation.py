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


def sim_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except SimulationError as exc:
        return exc
    raise AssertionError("SimulationError not raised")


class ExactExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_zero_state_z_expectation(self):
        result = self.svc.expectation({"qubit_count": 1}, ["Z"])
        self.assertEqual(result, {"mode": "exact", "results": [{"observable": "Z", "expectation": 1.0}]})

    def test_x_flips_z_expectation(self):
        result = self.svc.expectation({"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}, ["Z"])
        self.assertEqual(result["results"][0]["expectation"], -1.0)

    def test_hadamard_maps_z_to_x(self):
        result = self.svc.expectation({"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}, ["X", "Z"])
        self.assertAlmostEqual(result["results"][0]["expectation"], 1.0)
        self.assertAlmostEqual(result["results"][1]["expectation"], 0.0)

    def test_rx_uses_exp_minus_i_theta_p_over_2(self):
        # rx(pi) |0> = -i|1>；rx(pi/2) 时 <Y> = -sin(pi/2) = -1。
        circuit = {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": math.pi}]}
        result = self.svc.expectation(circuit, ["Z"])
        self.assertAlmostEqual(result["results"][0]["expectation"], -1.0)

        circuit["operations"][0]["angle"] = math.pi / 2
        result = self.svc.expectation(circuit, ["Y"])
        self.assertAlmostEqual(result["results"][0]["expectation"], -1.0)

    def test_rz_keeps_z_expectation(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "rz", "target": 0, "angle": 1.25}]}
        result = self.svc.expectation(circuit, ["Z"])
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_bell_state_correlations(self):
        circuit = {
            "qubit_count": 2,
            "operations": [{"gate": "h", "target": 0}, {"gate": "cx", "control": 0, "target": 1}],
        }
        result = self.svc.expectation(circuit, ["ZZ", "XX", "ZI", "IZ"])
        values = [entry["expectation"] for entry in result["results"]]
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[2], 0.0)
        self.assertAlmostEqual(values[3], 0.0)

    def test_observable_index_maps_to_qubit_number(self):
        # "XZ"：qubit 0 上为 X，qubit 1 上为 Z；x 作用于 qubit 1 后 <XZ> = 0。
        circuit = {"qubit_count": 2, "operations": [{"gate": "x", "target": 1}]}
        result = self.svc.expectation(circuit, ["IZ", "ZI"])
        self.assertEqual(result["results"][0]["expectation"], -1.0)
        self.assertEqual(result["results"][1]["expectation"], 1.0)

    def test_results_follow_input_order_and_json_native(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        observables = ["X", "Z", "X"]
        result = self.svc.expectation(circuit, observables)
        self.assertEqual([entry["observable"] for entry in result["results"]], observables)
        json.dumps(result, sort_keys=True)

    def test_empty_circuit_zero_qubits_identity_expectation(self):
        result = self.svc.expectation({"qubit_count": 0}, [""])
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_negative_zero_normalized(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.expectation(circuit, ["Z"])
        value = result["results"][0]["expectation"]
        self.assertEqual(value, 0.0)
        self.assertNotEqual(math.copysign(1.0, value), -1.0)

    def test_input_not_modified(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "theta"}}],
        }
        observables = ["Z"]
        values = {"theta": 0.5}
        snapshot = (copy.deepcopy(circuit), copy.deepcopy(observables), copy.deepcopy(values))
        self.svc.expectation(circuit, observables, values)
        self.assertEqual((circuit, observables, values), snapshot)

    def test_parameterized_circuit_with_values(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "theta"}}],
        }
        result = self.svc.expectation(circuit, ["Z"], {"theta": math.pi})
        self.assertAlmostEqual(result["results"][0]["expectation"], -1.0)

    def test_parameter_errors_propagate(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "theta"}}],
        }
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.expectation(circuit, ["Z"])
        self.assertEqual(ctx.exception.code, "missing_parameter")
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.expectation(circuit, ["Z"], {"theta": 1.0, "other": 2.0})
        self.assertEqual(ctx.exception.code, "unknown_parameter")

    def test_circuit_validation_errors_propagate(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.expectation({"qubit_count": 1, "operations": [{"gate": "x", "target": 2}]}, ["Z"])


class SampledExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_seed_makes_results_reproducible(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        first = self.svc.expectation(circuit, ["Z", "X"], shots=200, seed=42)
        second = self.svc.expectation(circuit, ["Z", "X"], shots=200, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "sampled")
        self.assertEqual(first["shots"], 200)

    def test_counts_sum_to_shots_and_match_expectation(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.expectation(circuit, ["Z"], shots=101, seed=7)
        entry = result["results"][0]
        counts = entry["counts"]
        self.assertEqual(counts["plus_one"] + counts["minus_one"], 101)
        self.assertEqual(entry["expectation"], (counts["plus_one"] - counts["minus_one"]) / 101)

    def test_certain_outcome_all_plus(self):
        result = self.svc.expectation({"qubit_count": 1}, ["Z"], shots=50, seed=1)
        self.assertEqual(result["results"][0]["counts"], {"plus_one": 50, "minus_one": 0})
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_empty_circuit_samples_all_plus(self):
        result = self.svc.expectation({"qubit_count": 0}, [""], shots=10, seed=3)
        self.assertEqual(result["results"][0]["counts"], {"plus_one": 10, "minus_one": 0})
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_sampling_is_statistically_consistent(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.expectation(circuit, ["Z"], shots=5000, seed=11)
        self.assertLess(abs(result["results"][0]["expectation"]), 0.1)

    def test_result_is_json_native(self):
        result = self.svc.expectation({"qubit_count": 1}, ["Z"], shots=10, seed=5)
        json.dumps(result, sort_keys=True)


class SimulationValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = {"qubit_count": 1}

    def test_observables_must_be_non_empty_array(self):
        for bad in (None, "Z", 1, {}, []):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, self.circuit, bad)
                self.assertEqual((e.code, e.path), ("invalid_observables", "observables"))

    def test_observable_items_must_be_valid_pauli_strings(self):
        for bad in (["z"], ["A"], ["ZZ"], [""], [1], [None], [["Z"]]):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, self.circuit, bad)
                self.assertEqual(e.code, "invalid_observable")
                self.assertEqual(e.path, "observables[0]")

    def test_observable_error_path_tracks_index(self):
        e = sim_err(self.svc.expectation, {"qubit_count": 2}, ["ZI", "Z"])
        self.assertEqual((e.code, e.path), ("invalid_observable", "observables[1]"))

    def test_shots_must_be_positive_integer(self):
        for bad in (0, -1, 1.5, "10", True, [1]):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, self.circuit, ["Z"], None, bad)
                self.assertEqual((e.code, e.path), ("invalid_shots", "shots"))

    def test_seed_must_be_non_negative_integer_in_sampled_mode(self):
        for bad in (-1, 1.5, "0", True):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, self.circuit, ["Z"], None, 10, bad)
                self.assertEqual((e.code, e.path), ("invalid_seed", "seed"))

    def test_seed_without_shots_rejected(self):
        e = sim_err(self.svc.expectation, self.circuit, ["Z"], None, None, 0)
        self.assertEqual((e.code, e.path), ("seed_without_shots", "seed"))

    def test_state_space_limit(self):
        e = sim_err(self.svc.expectation, {"qubit_count": 21}, ["I" * 21])
        self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))
        # 边界：20 个量子位合法。
        result = self.svc.expectation({"qubit_count": 20}, ["I" * 20])
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_error_has_stable_code_and_path(self):
        e = sim_err(self.svc.expectation, self.circuit, [])
        self.assertIsInstance(e, ValueError)
        self.assertEqual((e.code, e.path), ("invalid_observables", "observables"))


if __name__ == "__main__":
    unittest.main()
