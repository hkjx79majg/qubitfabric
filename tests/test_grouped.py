import copy
import json
import math
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


def bell_circuit():
    return {"qubit_count": 2, "operations": [
        {"gate": "h", "target": 0},
        {"gate": "cx", "control": 0, "target": 1},
    ]}


class GroupingStructureTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def group(self, observables):
        terms = [{"observable": obs, "coefficient": 1.0} for obs in observables]
        return self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 2}, terms, shots=100, seed=0,
        )["groups"]

    def test_first_fit_compatible_grouping(self):
        # XZ 建组；IZ 与 XZ 兼容并入；ZX 与第 0 组冲突新建；ZZ 两组都不兼容再建。
        groups = self.group(["XZ", "IZ", "ZX", "ZZ"])
        self.assertEqual(
            [g["basis"] for g in groups],
            ["XZ", "ZX", "ZZ"],
        )
        self.assertEqual([g["term_indexes"] for g in groups], [[0, 1], [2], [3]])

    def test_all_members_merge_into_first_group(self):
        groups = self.group(["ZI", "IZ", "ZZ", "II"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["basis"], "ZZ")
        self.assertEqual(groups[0]["term_indexes"], [0, 1, 2, 3])

    def test_incompatible_single_qubit_paulis_each_get_a_group(self):
        terms = [{"observable": obs, "coefficient": 1.0} for obs in ("X", "Y", "Z")]
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 1}, terms, shots=3, seed=0,
        )
        self.assertEqual([g["basis"] for g in result["groups"]], ["X", "Y", "Z"])
        self.assertEqual([g["term_indexes"] for g in result["groups"]], [[0], [1], [2]])

    def test_identity_first_shares_whatever_group_it_joins(self):
        # II 单独建空组，随后的 X 与它兼容并入，Z 冲突新建。
        groups = self.group(["II", "XI", "ZI"])
        self.assertEqual([g["basis"] for g in groups], ["XI", "ZI"])
        self.assertEqual([g["term_indexes"] for g in groups], [[0, 1], [2]])

    def test_share_one_basis_across_multiple_nonidentity_terms(self):
        groups = self.group(["XI", "IX", "XX"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["basis"], "XX")
        self.assertEqual(groups[0]["term_indexes"], [0, 1, 2])

    def test_identity_only_group_has_empty_basis(self):
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 0}, [{"observable": "", "coefficient": 2.5}], shots=1,
        )
        self.assertEqual(result["groups"], [{"basis": "", "shots": 1, "term_indexes": [0]}])


class BudgetAllocationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_shots_split_evenly_with_remainder_in_group_order(self):
        terms = [{"observable": obs, "coefficient": 1.0} for obs in ("X", "Y", "Z")]
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 1}, terms, shots=10, seed=0,
        )
        # 10 = 3*3 + 1：前一组 4，其余 3。
        self.assertEqual(result["shots"], 10)
        self.assertEqual([g["shots"] for g in result["groups"]], [4, 3, 3])
        self.assertEqual(sum(g["shots"] for g in result["groups"]), 10)

        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 1}, terms, shots=11, seed=0,
        )
        self.assertEqual([g["shots"] for g in result["groups"]], [4, 4, 3])

        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 1}, terms, shots=12, seed=0,
        )
        self.assertEqual([g["shots"] for g in result["groups"]], [4, 4, 4])

    def test_shots_equal_to_group_count_is_allowed(self):
        terms = [{"observable": obs, "coefficient": 1.0} for obs in ("X", "Y", "Z")]
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 1}, terms, shots=3, seed=0,
        )
        self.assertEqual([g["shots"] for g in result["groups"]], [1, 1, 1])

    def test_insufficient_shots(self):
        terms = [{"observable": obs, "coefficient": 1.0} for obs in ("X", "Y", "Z")]
        for bad in (1, 2):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.grouped_hamiltonian_expectation,
                            {"qubit_count": 1}, terms, shots=bad)
                self.assertEqual((e.code, e.path), ("insufficient_shots", "shots"))

    def test_counts_of_every_term_sum_to_its_group_shots(self):
        terms = [
            {"observable": "ZZ", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 2.0},
            {"observable": "XX", "coefficient": -1.0},
            {"observable": "II", "coefficient": 0.5},
        ]
        result = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, shots=99, seed=4)
        group_by_term = {}
        for g, group in enumerate(result["groups"]):
            self.assertGreaterEqual(group["shots"], 1)
            for index in group["term_indexes"]:
                group_by_term[index] = group["shots"]
        for i, item in enumerate(result["results"]):
            self.assertEqual(item["plus_count"] + item["minus_count"], group_by_term[i])
        self.assertEqual(sum(g["shots"] for g in result["groups"]), 99)


class JointSamplingTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_bell_correlations_from_joint_samples(self):
        terms = [
            {"observable": "ZZ", "coefficient": 1.0},
            {"observable": "XX", "coefficient": 0.5},
            {"observable": "ZI", "coefficient": 2.0},
            {"observable": "II", "coefficient": -3.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, shots=200, seed=7)
        by_obs = {item["observable"]: item for item in result["results"]}
        self.assertEqual(by_obs["ZZ"]["expectation"], 1.0)
        self.assertEqual(by_obs["ZZ"]["plus_count"], 100)
        self.assertEqual(by_obs["XX"]["expectation"], 1.0)
        self.assertEqual(by_obs["XX"]["plus_count"], 100)
        # 纯 I 项恒为 +1，计数沿用所属组的样本数。
        identity = by_obs["II"]
        self.assertEqual((identity["expectation"], identity["minus_count"]), (1.0, 0))
        self.assertEqual(identity["plus_count"], 100)
        # ZI 在 Bell 态上无信号，仅做统计粗检。
        self.assertAlmostEqual(by_obs["ZI"]["expectation"], 0.0, delta=0.3)
        energy = sum(item["coefficient"] * item["expectation"] for item in result["results"])
        self.assertEqual(result["energy"], energy)

    def test_duplicate_terms_are_preserved_and_share_identical_counts(self):
        terms = [
            {"observable": "ZZ", "coefficient": 1.0},
            {"observable": "ZZ", "coefficient": -2.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, shots=40, seed=3)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual([item["observable"] for item in result["results"]], ["ZZ", "ZZ"])
        self.assertEqual(result["groups"][0]["term_indexes"], [0, 1])
        self.assertEqual(result["results"][0]["plus_count"], result["results"][1]["plus_count"])
        self.assertEqual(result["energy"], -1.0)

    def test_x_and_y_basis_rotations_are_correct(self):
        # H 后 X 本征态：X 恒为 +1。
        h_circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.grouped_hamiltonian_expectation(
            h_circuit,
            [{"observable": "X", "coefficient": 1.0},
             {"observable": "Z", "coefficient": 1.0}],
            shots=100, seed=0,
        )
        x_item, z_item = result["results"]
        self.assertEqual((x_item["expectation"], x_item["minus_count"]), (1.0, 0))
        self.assertAlmostEqual(z_item["expectation"], 0.0, delta=0.25)

        # rx(π/2)|0> 是 Y 的 -1 本征态（exp(-iθX/2) 约定）。
        rx_circuit = {"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": math.pi / 2},
        ]}
        result = self.svc.grouped_hamiltonian_expectation(
            rx_circuit, [{"observable": "Y", "coefficient": 1.0}], shots=64, seed=2,
        )
        item = result["results"][0]
        self.assertEqual((item["expectation"], item["plus_count"]), (-1.0, 0))
        self.assertEqual(item["minus_count"], 64)

    def test_parameters_are_bound(self):
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        result = self.svc.grouped_hamiltonian_expectation(
            circuit, [{"observable": "Z", "coefficient": 1.0}],
            values={"theta": math.pi}, shots=50, seed=0,
        )
        self.assertEqual(result["results"][0]["expectation"], -1.0)

    def test_same_seed_reproduces_exactly_and_omitted_seed_is_deterministic(self):
        terms = [{"observable": obs, "coefficient": 1.0}
                 for obs in ("ZZ", "XX", "ZI", "YY")]
        kwargs = dict(shots=300)
        first = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, seed=42, **kwargs)
        second = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, seed=42, **kwargs)
        self.assertEqual(first, second)
        omitted_a = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, **kwargs)
        omitted_b = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, **kwargs)
        self.assertEqual(omitted_a, omitted_b)
        self.assertEqual(omitted_a, self.svc.grouped_hamiltonian_expectation(
            bell_circuit(), terms, seed=0, **kwargs))

    def test_different_seeds_usually_differ(self):
        terms = [{"observable": "ZI", "coefficient": 1.0}]
        a = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, shots=1000, seed=1)
        b = self.svc.grouped_hamiltonian_expectation(bell_circuit(), terms, shots=1000, seed=2)
        self.assertNotEqual(a["results"][0]["plus_count"], b["results"][0]["plus_count"])

    def test_group_random_streams_are_isolated_by_group_index(self):
        # 同一电路与测量基下：单项 X、100 shots 与 [X, Y, Z]、300 shots 的
        # 第 0 组同为 basis X、100 shots，随机流只由 seed 与组序号决定，
        # X 计数必须逐值一致；后两个组各用自己的派生流。
        h_circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        one = self.svc.grouped_hamiltonian_expectation(
            h_circuit, [{"observable": "X", "coefficient": 1.0}],
            shots=100, seed=9,
        )
        three = self.svc.grouped_hamiltonian_expectation(
            h_circuit,
            [{"observable": "X", "coefficient": 1.0},
             {"observable": "Y", "coefficient": 1.0},
             {"observable": "Z", "coefficient": 1.0}],
            shots=300, seed=9,
        )
        self.assertEqual([g["shots"] for g in three["groups"]], [100, 100, 100])
        self.assertEqual(
            one["results"][0]["plus_count"],
            three["results"][0]["plus_count"],
        )
        # X 本征态上 X 恒为 +1，两组流隔离不影响该确定性结果。
        self.assertEqual(three["results"][0]["minus_count"], 0)

    def test_sampling_is_statistically_consistent(self):
        # H|0> 上 Z 期望为 0；大量 shots 下应落在附近。
        h_circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.svc.grouped_hamiltonian_expectation(
            h_circuit, [{"observable": "Z", "coefficient": 1.0}], shots=20000, seed=11,
        )
        self.assertAlmostEqual(result["results"][0]["expectation"], 0.0, delta=0.05)


class NoiseTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_noisy_grouped_matches_depolarized_correlations(self):
        noise = {"two_qubit_depolarizing": 0.25}
        terms = [
            {"observable": "ZZ", "coefficient": 1.0},
            {"observable": "XX", "coefficient": 1.0},
            {"observable": "ZI", "coefficient": 1.0},
            {"observable": "II", "coefficient": 1.0},
        ]
        result = self.svc.grouped_hamiltonian_expectation(
            bell_circuit(), terms, shots=4000, seed=5, noise=noise,
        )
        by_obs = {item["observable"]: item for item in result["results"]}
        self.assertAlmostEqual(by_obs["ZZ"]["expectation"], 0.75, delta=0.08)
        self.assertAlmostEqual(by_obs["XX"]["expectation"], 0.75, delta=0.08)
        self.assertAlmostEqual(by_obs["ZI"]["expectation"], 0.0, delta=0.1)
        self.assertEqual(by_obs["II"]["expectation"], 1.0)

    def test_zero_and_empty_noise_match_omitted(self):
        terms = [{"observable": "ZZ", "coefficient": 1.0},
                 {"observable": "XI", "coefficient": 0.5}]
        kwargs = dict(values={"t": 0.9}, shots=80, seed=3)
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        baseline = self.svc.grouped_hamiltonian_expectation(circuit, terms, **kwargs)
        for noise in (None, {},
                      {"single_qubit_depolarizing": 0.0},
                      {"single_qubit_depolarizing": 0, "two_qubit_depolarizing": 0}):
            with self.subTest(noise=noise):
                got = self.svc.grouped_hamiltonian_expectation(circuit, terms, noise=noise, **kwargs)
                self.assertEqual(got, baseline)

    def test_noisy_qubit_limit_is_ten(self):
        e = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            {"qubit_count": 11},
            [{"observable": "I" * 11, "coefficient": 1.0}],
            shots=20, noise={"single_qubit_depolarizing": 0.1},
        )
        self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 11},
            [{"observable": "I" * 11, "coefficient": 1.0}],
            shots=20, noise={"single_qubit_depolarizing": 0.0},
        )
        self.assertEqual(result["results"][0]["expectation"], 1.0)
        result = self.svc.grouped_hamiltonian_expectation(
            {"qubit_count": 20},
            [{"observable": "I" * 20, "coefficient": 1.0}],
            shots=20,
        )
        self.assertEqual(result["results"][0]["expectation"], 1.0)


class GroupedValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_circuit_and_binding_validated_first(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.grouped_hamiltonian_expectation(
                {"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]},
                [{"observable": "Z", "coefficient": 1.0}], shots=10,
            )
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        with self.assertRaises(ParameterBindingError):
            self.svc.grouped_hamiltonian_expectation(
                circuit, [{"observable": "Z", "coefficient": 1.0}], shots=10,
            )
        with self.assertRaises(ParameterBindingError):
            self.svc.grouped_hamiltonian_expectation(
                {"qubit_count": 1}, [{"observable": "Z", "coefficient": 1.0}],
                values={"nope": 1.0}, shots=10,
            )

    def test_term_structure_reuses_optimization_error(self):
        cases = [
            ([], "invalid_terms", "terms"),
            ("x", "invalid_terms", "terms"),
            ([1], "invalid_term", "terms[0]"),
            ([{"coefficient": 1.0}], "invalid_term", "terms[0].observable"),
            ([{"observable": 1, "coefficient": 1.0}], "invalid_term", "terms[0].observable"),
            ([{"observable": "Z"}], "invalid_term", "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": "x"}], "invalid_term", "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": True}], "invalid_term", "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": math.inf}],
             "invalid_term", "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": 1.0, "extra": 1}],
             "invalid_term", "terms[0].extra"),
        ]
        for terms, code, path in cases:
            with self.subTest(terms=terms):
                e = opt_err(self.svc.grouped_hamiltonian_expectation,
                            {"qubit_count": 1}, terms, shots=10)
                self.assertEqual((e.code, e.path), (code, path))

    def test_observable_content_reuses_simulation_error(self):
        cases = [
            ([{"observable": "ZZ", "coefficient": 1.0}], "observables[0]"),
            ([{"observable": "z", "coefficient": 1.0}], "observables[0]"),
            ([{"observable": "A", "coefficient": 1.0}], "observables[0]"),
        ]
        for terms, path in cases:
            with self.subTest(terms=terms):
                e = sim_err(self.svc.grouped_hamiltonian_expectation,
                            {"qubit_count": 1}, terms, shots=10)
                self.assertEqual((e.code, e.path), ("invalid_observable", path))

    def test_noise_validated_before_shots_and_after_observable(self):
        e = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            {"qubit_count": 1},
            [{"observable": "Z", "coefficient": 1.0}],
            shots="bad", noise="bad",
        )
        self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise"))
        e = sim_err(
            self.svc.grouped_hamiltonian_expectation,
            {"qubit_count": 1},
            [{"observable": "ZZ", "coefficient": 1.0}],
            shots=10, noise="bad",
        )
        self.assertEqual((e.code, e.path), ("invalid_observable", "observables[0]"))

    def test_invalid_shots(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        for bad in (None, 0, -3, 1.5, "10", True, []):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.grouped_hamiltonian_expectation,
                            {"qubit_count": 1}, terms, shots=bad)
                self.assertEqual((e.code, e.path), ("invalid_shots", "shots"))

    def test_invalid_seed(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        for bad in (-1, 0.5, "0", True, 2.0):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.grouped_hamiltonian_expectation,
                            {"qubit_count": 1}, terms, shots=10, seed=bad)
                self.assertEqual((e.code, e.path), ("invalid_seed", "seed"))

    def test_insufficient_shots_precedes_seed_validation(self):
        terms = [{"observable": obs, "coefficient": 1.0} for obs in ("X", "Y", "Z")]
        e = sim_err(self.svc.grouped_hamiltonian_expectation,
                    {"qubit_count": 1}, terms, shots=2, seed=-1)
        self.assertEqual((e.code, e.path), ("insufficient_shots", "shots"))


class OutputContractTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_inputs_not_mutated_and_result_is_json_native(self):
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rz", "target": 1, "angle": {"parameter": "t"}},
        ]}
        terms = [
            {"observable": "ZZ", "coefficient": 1.0},
            {"observable": "XI", "coefficient": -0.5},
            {"observable": "II", "coefficient": 3.0},
        ]
        noise = {"single_qubit_depolarizing": 0.1, "two_qubit_depolarizing": 0.05}
        values = {"t": 0.3}
        snapshot = (copy.deepcopy(circuit), copy.deepcopy(terms),
                    copy.deepcopy(noise), copy.deepcopy(values))
        result = self.svc.grouped_hamiltonian_expectation(
            circuit, terms, values=values, shots=70, seed=1, noise=noise,
        )
        self.assertEqual((circuit, terms, noise, values), snapshot)
        text = json.dumps(result, sort_keys=True)
        self.assertNotIn("-0.0", text)
        self.assertEqual(set(result), {"qubit_count", "shots", "groups", "results", "energy"})
        for group in result["groups"]:
            self.assertEqual(set(group), {"basis", "shots", "term_indexes"})
        for item in result["results"]:
            self.assertEqual(set(item),
                             {"observable", "coefficient", "expectation",
                              "plus_count", "minus_count"})

    def test_results_align_one_to_one_with_terms(self):
        terms = [
            {"observable": "ZI", "coefficient": 2.0},
            {"observable": "IZ", "coefficient": -1.5},
            {"observable": "ZI", "coefficient": 0.0},
        ]
        circuit = {"qubit_count": 2, "operations": [{"gate": "x", "target": 1}]}
        result = self.svc.grouped_hamiltonian_expectation(circuit, terms, shots=30, seed=0)
        self.assertEqual([item["observable"] for item in result["results"]],
                         ["ZI", "IZ", "ZI"])
        self.assertEqual([item["coefficient"] for item in result["results"]],
                         [2.0, -1.5, 0.0])
        # x 作用在 qubit 1：ZI=+1、IZ=-1；energy = 2*1 + -1.5*-1 + 0 = 3.5。
        self.assertEqual(result["energy"], 3.5)


if __name__ == "__main__":
    unittest.main()
