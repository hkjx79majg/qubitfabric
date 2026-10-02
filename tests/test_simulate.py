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


def bind_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ParameterBindingError as exc:
        return exc
    raise AssertionError("ParameterBindingError not raised")


class ExactExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def expectations(self, circuit, observables, values=None):
        result = self.svc.expectation(circuit, observables, values=values)
        return [item["expectation"] for item in result["results"]]

    def test_ground_state_z_and_identity(self):
        values = self.expectations({"qubit_count": 1}, ["Z", "I", "X", "Y"])
        self.assertEqual(values, [1.0, 1.0, 0.0, 0.0])

    def test_x_flips_z_expectation(self):
        values = self.expectations({"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}, ["Z"])
        self.assertEqual(values, [-1.0])

    def test_qubit_zero_is_least_significant_and_string_index_matches_qubit(self):
        circuit = {"qubit_count": 2, "operations": [{"gate": "x", "target": 0}]}
        # 下标 0 对应 qubit 0：ZI 测 qubit 0 的 Z，IZ 测 qubit 1 的 Z。
        values = self.expectations(circuit, ["ZI", "IZ"])
        self.assertEqual(values, [-1.0, 1.0])

    def test_hadamard_creates_x_eigenstate(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        values = self.expectations(circuit, ["X", "Z"])
        self.assertAlmostEqual(values[0], 1.0, places=12)
        self.assertAlmostEqual(values[1], 0.0, places=12)

    def test_rx_uses_exp_minus_i_theta_p_over_2(self):
        # rx(pi)|0> = -i|1>，Z 期望为 -1。
        circuit = {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": math.pi}]}
        self.assertAlmostEqual(self.expectations(circuit, ["Z"])[0], -1.0, places=12)
        # rx(pi/2)|0> 的 Y 期望为 -1（exp(-iθX/2) 约定）。
        circuit = {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": math.pi / 2}]}
        self.assertAlmostEqual(self.expectations(circuit, ["Y"])[0], -1.0, places=12)

    def test_rz_rotates_plus_state_in_xy_plane(self):
        theta = 0.7
        circuit = {"qubit_count": 1, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "rz", "target": 0, "angle": theta},
        ]}
        values = self.expectations(circuit, ["X", "Y"])
        self.assertAlmostEqual(values[0], math.cos(theta), places=12)
        self.assertAlmostEqual(values[1], math.sin(theta), places=12)

    def test_bell_state_correlations(self):
        circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        values = self.expectations(circuit, ["ZZ", "XX", "ZI", "IZ", "YY"])
        self.assertAlmostEqual(values[0], 1.0, places=12)
        self.assertAlmostEqual(values[1], 1.0, places=12)
        self.assertAlmostEqual(values[2], 0.0, places=12)
        self.assertAlmostEqual(values[3], 0.0, places=12)
        self.assertAlmostEqual(values[4], -1.0, places=12)

    def test_results_follow_observable_order(self):
        result = self.svc.expectation({"qubit_count": 1}, ["Z", "X", "Z"])
        self.assertEqual([r["observable"] for r in result["results"]], ["Z", "X", "Z"])
        self.assertEqual(result["shots"], None)
        self.assertEqual(result["qubit_count"], 1)

    def test_empty_circuit_empty_pauli_string(self):
        result = self.svc.expectation({"qubit_count": 0}, [""])
        self.assertEqual(result["results"], [{"observable": "", "expectation": 1.0}])

    def test_parameterized_circuit_with_values(self):
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        values = self.expectations(circuit, ["Z"], values={"theta": math.pi})
        self.assertAlmostEqual(values[0], -1.0, places=12)

    def test_no_parameter_circuit_allows_omitted_or_empty_values(self):
        circuit = {"qubit_count": 1}
        self.assertEqual(
            self.svc.expectation(circuit, ["Z"]),
            self.svc.expectation(circuit, ["Z"], values={}),
        )

    def test_missing_and_unknown_parameter_raise_binding_error(self):
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        e = bind_err(self.svc.expectation, circuit, ["Z"])
        self.assertEqual((e.code, e.path), ("missing_parameter", "theta"))
        e = bind_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], values={"nope": 1})
        self.assertEqual((e.code, e.path), ("unknown_parameter", "nope"))

    def test_circuit_validation_is_reused(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.expectation({"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]}, ["Z"])

    def test_input_not_mutated_and_result_json_serializable(self):
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        observables = ["Z", "X"]
        snapshot = (copy.deepcopy(circuit), list(observables))
        result = self.svc.expectation(circuit, observables, values={"t": 0.5}, shots=10, seed=1)
        self.assertEqual((circuit, observables), snapshot)
        json.dumps(result, sort_keys=True)
        for item in result["results"]:
            self.assertTrue(-1.0 <= item["expectation"] <= 1.0)


class SampledExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_counts_sum_to_shots_and_expectation_matches(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.expectation(circuit, ["X", "Z"], shots=200, seed=7)
        self.assertEqual(result["shots"], 200)
        for item in result["results"]:
            counts = item["counts"]
            self.assertEqual(counts["positive"] + counts["negative"], 200)
            expected = (counts["positive"] - counts["negative"]) / 200
            self.assertEqual(item["expectation"], expected if expected != 0 else 0.0)
        # X 本征态上采样必全部为正一。
        self.assertEqual(result["results"][0]["counts"], {"positive": 200, "negative": 0})

    def test_same_seed_reproduces_exactly(self):
        circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rz", "target": 1, "angle": 0.3},
        ]}
        args = (circuit, ["ZZ", "XX", "YY"])
        first = self.svc.expectation(*args, shots=500, seed=42)
        second = self.svc.expectation(*args, shots=500, seed=42)
        self.assertEqual(first, second)

    def test_different_seeds_usually_differ(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        a = self.svc.expectation(circuit, ["Z"], shots=1000, seed=1)
        b = self.svc.expectation(circuit, ["Z"], shots=1000, seed=2)
        self.assertNotEqual(a["results"][0]["counts"], b["results"][0]["counts"])

    def test_omitted_seed_is_deterministic(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        self.assertEqual(
            self.svc.expectation(circuit, ["Z"], shots=100),
            self.svc.expectation(circuit, ["Z"], shots=100),
        )

    def test_empty_pauli_string_samples_all_positive(self):
        result = self.svc.expectation({"qubit_count": 0}, [""], shots=25, seed=3)
        self.assertEqual(result["results"][0]["counts"], {"positive": 25, "negative": 0})
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_sampling_is_statistically_consistent(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.expectation(circuit, ["Z"], shots=20000, seed=11)
        self.assertAlmostEqual(result["results"][0]["expectation"], 0.0, delta=0.05)


class SimulationValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_observables_must_be_non_empty_array(self):
        for bad in (None, "Z", {"0": "Z"}, [], 1):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, {"qubit_count": 1}, bad)
                self.assertEqual((e.code, e.path), ("invalid_observables", "observables"))

    def test_invalid_observable_items(self):
        cases = [
            (["Z", 1], "observables[1]"),
            ([None], "observables[0]"),
            (["ZZ"], "observables[0]"),
            ([""], "observables[0]"),
            (["z"], "observables[0]"),
            (["A"], "observables[0]"),
        ]
        for observables, path in cases:
            with self.subTest(observables=observables):
                e = sim_err(self.svc.expectation, {"qubit_count": 1}, observables)
                self.assertEqual((e.code, e.path), ("invalid_observable", path))

    def test_invalid_shots(self):
        for bad in (0, -3, 1.5, "10", True, []):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], shots=bad)
                self.assertEqual((e.code, e.path), ("invalid_shots", "shots"))

    def test_invalid_seed(self):
        for bad in (-1, 0.5, "0", True, 2.0):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], shots=10, seed=bad)
                self.assertEqual((e.code, e.path), ("invalid_seed", "seed"))

    def test_seed_without_shots(self):
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], seed=0)
        self.assertEqual((e.code, e.path), ("seed_without_shots", "seed"))

    def test_state_space_too_large(self):
        observables = ["I" * 21]
        e = sim_err(self.svc.expectation, {"qubit_count": 21}, observables)
        self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))
        # 边界：20 个量子位仍然可用。
        result = self.svc.expectation({"qubit_count": 20}, ["I" * 20])
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_error_is_importable_and_stable(self):
        from qubitfabric.service import SimulationError as FromService
        from qubitfabric import SimulationError as FromPackage
        self.assertIs(FromService, FromPackage)
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, [])
        self.assertIsInstance(e, ValueError)
        self.assertEqual((e.code, e.path), ("invalid_observables", "observables"))


if __name__ == "__main__":
    unittest.main()
