import copy
import json
import unittest

from qubitfabric.service import (
    CircuitValidationError,
    OptimizationError,
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


def opt_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except OptimizationError as exc:
        return exc
    raise AssertionError("OptimizationError not raised")


class GroupedHamiltonianTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.bell = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}

    def test_grouping_earliest_compatible_and_basis_merge(self):
        terms = [
            {"observable": "XI", "coefficient": 1.0},
            {"observable": "IX", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 1.0},
            {"observable": "XX", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 2, "operations": []}, terms, shots=11, seed=7,
        )
        self.assertEqual(
            result["groups"],
            [
                {"basis": "XX", "shots": 6, "term_indexes": [0, 1, 3]},
                {"basis": "ZI", "shots": 5, "term_indexes": [2]},
            ],
        )
        self.assertEqual(result["shots"], 11)
        self.assertEqual(result["qubit_count"], 2)
        self.assertEqual(len(result["results"]), 4)

    def test_incompatible_terms_create_new_groups(self):
        terms = [
            {"observable": "XI", "coefficient": 1.0},
            {"observable": "YI", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 2, "operations": []}, terms, shots=9, seed=1,
        )
        self.assertEqual([g["basis"] for g in result["groups"]], ["XI", "YI", "ZI"])
        self.assertEqual([g["shots"] for g in result["groups"]], [3, 3, 3])

    def test_shots_remainder_goes_to_earliest_groups(self):
        terms = [
            {"observable": "XI", "coefficient": 1.0},
            {"observable": "YI", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 2, "operations": []}, terms, shots=10, seed=1,
        )
        self.assertEqual([g["shots"] for g in result["groups"]], [4, 3, 3])

    def test_insufficient_shots(self):
        terms = [
            {"observable": "XI", "coefficient": 1.0},
            {"observable": "YI", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 1.0},
        ]
        exc = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            {"qubit_count": 2, "operations": []}, terms, shots=2,
        )
        self.assertEqual(exc.code, "insufficient_shots")
        self.assertEqual(exc.path, "shots")

    def test_counts_sum_to_group_shots(self):
        terms = [
            {"observable": "XI", "coefficient": 1.0},
            {"observable": "YI", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(
            self.bell, terms, shots=30, seed=5,
        )
        for item, group in zip(result["results"], result["groups"]):
            self.assertEqual(item["plus_count"] + item["minus_count"], group["shots"])

    def test_shared_bitstrings_give_consistent_counts(self):
        # Bell 态上 ZI 与 IZ 每个 bitstring 同号，ZZ 恒为二者乘积 +1。
        terms = [
            {"observable": "ZI", "coefficient": 1.0},
            {"observable": "IZ", "coefficient": 1.0},
            {"observable": "ZZ", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(self.bell, terms, shots=200, seed=3)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["basis"], "ZZ")
        first, second, product = result["results"]
        self.assertEqual(first["plus_count"], second["plus_count"])
        self.assertEqual(first["minus_count"], second["minus_count"])
        self.assertEqual(product["plus_count"], 200)
        self.assertEqual(product["minus_count"], 0)
        self.assertEqual(product["expectation"], 1.0)

    def test_pure_identity_term_is_always_plus_one(self):
        terms = [
            {"observable": "II", "coefficient": 2.5},
            {"observable": "ZZ", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(self.bell, terms, shots=50, seed=2)
        identity = result["results"][0]
        self.assertEqual(identity["expectation"], 1.0)
        self.assertEqual(identity["minus_count"], 0)
        self.assertEqual(identity["plus_count"], result["groups"][0]["shots"])
        # 纯 I 项并入最早兼容组，不独占预算。
        self.assertEqual(result["groups"][0]["term_indexes"], [0, 1])

    def test_deterministic_basis_is_exact(self):
        # |+> 上 X 的测量结果恒为 +1，计数与期望不依赖随机性。
        terms = [{"observable": "X", "coefficient": 3.0}]
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.grouped_hamiltonian_expectation(circuit, terms, shots=100, seed=9)
        item = result["results"][0]
        self.assertEqual(item["plus_count"], 100)
        self.assertEqual(item["minus_count"], 0)
        self.assertEqual(item["expectation"], 1.0)
        self.assertEqual(result["energy"], 3.0)

    def test_y_basis_rotation(self):
        # rx 不适用；用 h+rz 构造 Y 本征态：S|0> = (|0>+i|1>)/√2 经 h、rz(π/2)。
        circuit = {"qubit_count": 1, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "rz", "target": 0, "angle": 1.5707963267948966},
        ]}
        terms = [{"observable": "Y", "coefficient": 1.0}]
        result = self.svc.grouped_hamiltonian_expectation(circuit, terms, shots=100, seed=4)
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_sampling_matches_exact_probability_statistically(self):
        # |0> 上 Z 恒为 +1；h|0> 上 Z 为 50/50，大样本下接近期望。
        terms = [{"observable": "Z", "coefficient": 1.0}]
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.grouped_hamiltonian_expectation(circuit, terms, shots=20000, seed=11)
        self.assertAlmostEqual(result["results"][0]["expectation"], 0.0, delta=0.05)

    def test_same_seed_reproduces_results(self):
        terms = [
            {"observable": "XX", "coefficient": 1.5},
            {"observable": "ZI", "coefficient": -0.5},
            {"observable": "IZ", "coefficient": 0.25},
        ]
        kwargs = dict(shots=500, seed=42)
        first = self.svc.grouped_hamiltonian_expectation(self.bell, terms, **kwargs)
        second = self.svc.grouped_hamiltonian_expectation(self.bell, terms, **kwargs)
        self.assertEqual(first, second)

    def test_seed_isolates_group_streams(self):
        terms = [
            {"observable": "XI", "coefficient": 1.0},
            {"observable": "YI", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(self.bell, terms, shots=200, seed=8)
        again = self.svc.grouped_hamiltonian_expectation(self.bell, terms, shots=200, seed=9)
        self.assertNotEqual(
            [r["plus_count"] for r in result["results"]],
            [r["plus_count"] for r in again["results"]],
        )

    def test_energy_sums_in_input_order_with_duplicates(self):
        terms = [
            {"observable": "ZZ", "coefficient": 2.0},
            {"observable": "ZZ", "coefficient": 0.5},
            {"observable": "ZI", "coefficient": -1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(self.bell, terms, shots=100, seed=6)
        energy = 0.0
        for item in result["results"]:
            energy += item["coefficient"] * item["expectation"]
        self.assertEqual(result["energy"], energy)
        # 重复项保留且共享同组样本，逐值一致。
        self.assertEqual(result["results"][0]["expectation"], result["results"][1]["expectation"])
        self.assertEqual(result["results"][0]["plus_count"], result["results"][1]["plus_count"])

    def test_zero_noise_matches_omitted_noise(self):
        terms = [
            {"observable": "XX", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 1.0},
        ]
        plain = self.svc.grouped_hamiltonian_expectation(self.bell, terms, shots=300, seed=5)
        zero = self.svc.grouped_hamiltonian_expectation(
            self.bell, terms, shots=300, seed=5,
            noise={"single_qubit_depolarizing": 0.0, "two_qubit_depolarizing": 0.0},
        )
        self.assertEqual(plain, zero)

    def test_noisy_sampling_uses_noisy_state(self):
        # p=1 单比特退极化把任何单比特态完全混合，Z 期望统计上为 0。
        circuit = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        terms = [{"observable": "Z", "coefficient": 1.0}]
        result = self.svc.grouped_hamiltonian_expectation(
            circuit, terms, shots=20000, seed=13,
            noise={"single_qubit_depolarizing": 1.0},
        )
        self.assertAlmostEqual(result["results"][0]["expectation"], 0.0, delta=0.05)

    def test_parameter_binding(self):
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "theta"}}],
        }
        terms = [{"observable": "Z", "coefficient": 1.0}]
        result = self.svc.grouped_hamiltonian_expectation(
            circuit, terms, values={"theta": 0.0}, shots=50, seed=1,
        )
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_results_are_json_native_and_input_untouched(self):
        circuit = copy.deepcopy(self.bell)
        terms = [
            {"observable": "XX", "coefficient": 1},
            {"observable": "ZI", "coefficient": -0.5},
        ]
        terms_snapshot = copy.deepcopy(terms)
        circuit_snapshot = copy.deepcopy(circuit)
        result = self.svc.grouped_hamiltonian_expectation(
            circuit, terms, shots=100, seed=1,
        )
        json.dumps(result)
        self.assertEqual(circuit, circuit_snapshot)
        self.assertEqual(terms, terms_snapshot)
        # 系数规范为 JSON 数值。
        self.assertEqual(result["results"][0]["coefficient"], 1.0)


class GroupedHamiltonianValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = {"qubit_count": 1, "operations": []}
        self.terms = [{"observable": "Z", "coefficient": 1.0}]

    def test_invalid_shots_values(self):
        for bad in (None, 0, -3, 1.5, True, "10"):
            with self.subTest(shots=bad):
                exc = sim_err(
                    self.svc.grouped_hamiltonian_expectation,
                    self.circuit, self.terms, shots=bad,
                )
                self.assertEqual(exc.code, "invalid_shots")
                self.assertEqual(exc.path, "shots")

    def test_invalid_seed(self):
        for bad in (-1, 1.5, True, "x"):
            with self.subTest(seed=bad):
                exc = sim_err(
                    self.svc.grouped_hamiltonian_expectation,
                    self.circuit, self.terms, shots=10, seed=bad,
                )
                self.assertEqual(exc.code, "invalid_seed")
                self.assertEqual(exc.path, "seed")

    def test_terms_structure_reuses_optimization_error(self):
        exc = opt_err(self.svc.grouped_hamiltonian_expectation, self.circuit, [], shots=10)
        self.assertEqual(exc.code, "invalid_terms")
        self.assertEqual(exc.path, "terms")
        exc = opt_err(
            self.svc.grouped_hamiltonian_expectation,
            self.circuit, [{"observable": "Z"}], shots=10,
        )
        self.assertEqual(exc.code, "invalid_term")
        self.assertEqual(exc.path, "terms[0].coefficient")
        exc = opt_err(
            self.svc.grouped_hamiltonian_expectation,
            self.circuit, [{"observable": "Z", "coefficient": float("nan")}], shots=10,
        )
        self.assertEqual(exc.code, "invalid_term")
        self.assertEqual(exc.path, "terms[0].coefficient")

    def test_observable_content_reuses_simulation_error(self):
        exc = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            self.circuit, [{"observable": "ZZ", "coefficient": 1.0}], shots=10,
        )
        self.assertEqual(exc.code, "invalid_observable")
        self.assertEqual(exc.path, "observables[0]")
        exc = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            self.circuit, [{"observable": "A", "coefficient": 1.0}], shots=10,
        )
        self.assertEqual(exc.code, "invalid_observable")

    def test_noise_validation_reused(self):
        exc = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            self.circuit, self.terms, shots=10,
            noise={"single_qubit_depolarizing": 1.5},
        )
        self.assertEqual(exc.code, "invalid_noise_model")
        self.assertEqual(exc.path, "noise.single_qubit_depolarizing")

    def test_circuit_and_binding_errors_unchanged(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.grouped_hamiltonian_expectation(
                {"operations": []}, self.terms, shots=10,
            )
        circuit = {
            "qubit_count": 1,
            "parameters": ["theta"],
            "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "theta"}}],
        }
        with self.assertRaises(ParameterBindingError):
            self.svc.grouped_hamiltonian_expectation(circuit, self.terms, shots=10)

    def test_validation_order(self):
        # 电路错误优先于 shots 错误。
        with self.assertRaises(CircuitValidationError):
            self.svc.grouped_hamiltonian_expectation({}, self.terms, shots=0)
        # terms 结构错误优先于 shots 错误。
        exc = opt_err(self.svc.grouped_hamiltonian_expectation, self.circuit, "x", shots=0)
        self.assertEqual(exc.code, "invalid_terms")
        # shots 错误优先于 seed 错误。
        exc = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            self.circuit, self.terms, shots=0, seed=-1,
        )
        self.assertEqual(exc.code, "invalid_shots")

    def test_state_space_limit_reused(self):
        circuit = {"qubit_count": 21, "operations": []}
        terms = [{"observable": "I" * 21, "coefficient": 1.0}]
        exc = sim_err(self.svc.grouped_hamiltonian_expectation, circuit, terms, shots=10)
        self.assertEqual(exc.code, "state_space_too_large")
        self.assertEqual(exc.path, "qubit_count")


if __name__ == "__main__":
    unittest.main()
