import copy
import json
import math
import unittest

from qubitfabric.service import CircuitValidationError, Service, SimulationError


def sim_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except SimulationError as exc:
        return exc
    raise AssertionError("SimulationError not raised")


class NoisyExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def expectation(self, circuit, observable, noise):
        return self.svc.expectation(circuit, [observable], noise=noise)["results"][0]["expectation"]

    def test_single_qubit_depolarizing_scales_bloch_vector(self):
        p = 0.3
        noise = {"single_qubit_depolarizing": p}
        flipped = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        self.assertAlmostEqual(self.expectation(flipped, "Z", noise), -(1 - p), places=12)
        plus = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        self.assertAlmostEqual(self.expectation(plus, "X", noise), 1 - p, places=12)
        self.assertEqual(self.expectation(plus, "Z", noise), 0.0)

    def test_rx_with_noise_matches_analytic(self):
        p = 0.3
        theta = 0.7
        circuit = {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": theta}]}
        value = self.expectation(circuit, "Z", {"single_qubit_depolarizing": p})
        self.assertAlmostEqual(value, (1 - p) * math.cos(theta), places=12)

    def test_full_depolarizing_gives_zero(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        noise = {"single_qubit_depolarizing": 1}
        self.assertEqual(self.expectation(circuit, "Z", noise), 0.0)

    def test_channel_applied_after_every_gate_in_order(self):
        # h、x 之后各施加一次通道：|+> 是 x 的不动点，<X> = (1-p)^2。
        p = 0.35
        circuit = {"qubit_count": 1, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "x", "target": 0},
        ]}
        value = self.expectation(circuit, "X", {"single_qubit_depolarizing": p})
        self.assertAlmostEqual(value, (1 - p) ** 2, places=12)

    def test_two_qubit_depolarizing_scales_bell_correlations(self):
        p = 0.25
        bell = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        noise = {"two_qubit_depolarizing": p}
        self.assertAlmostEqual(self.expectation(bell, "ZZ", noise), 1 - p, places=12)
        self.assertAlmostEqual(self.expectation(bell, "XX", noise), 1 - p, places=12)
        self.assertAlmostEqual(self.expectation(bell, "YY", noise), -(1 - p), places=12)
        self.assertEqual(self.expectation(bell, "ZI", noise), 0.0)

    def test_qubit_string_index_matches_qubit_in_noisy_sim(self):
        circuit = {"qubit_count": 3, "operations": [{"gate": "x", "target": 2}]}
        noise = {"single_qubit_depolarizing": 0.4}
        self.assertAlmostEqual(self.expectation(circuit, "IIZ", noise), -(1 - 0.4), places=12)
        # qubit 0/1 未被门触及，也不受噪声影响。
        self.assertEqual(self.expectation(circuit, "ZII", noise), 1.0)

    def test_both_noise_channels_compose(self):
        p1, p2 = 0.1, 0.2
        circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        noise = {"single_qubit_depolarizing": p1, "two_qubit_depolarizing": p2}
        # h 之后 qubit 0 退极化，cx 后两位退极化；XX 关联 = (1-p1)(1-p2)。
        value = self.expectation(circuit, "XX", noise)
        self.assertAlmostEqual(value, (1 - p1) * (1 - p2), places=12)

    def test_zero_and_empty_noise_match_baseline(self):
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        baseline = self.svc.expectation(circuit, ["ZZ", "XI"], values={"t": 0.9}, shots=50, seed=3)
        for noise in (None, {}, {"single_qubit_depolarizing": 0.0},
                      {"single_qubit_depolarizing": 0, "two_qubit_depolarizing": 0.0}):
            with self.subTest(noise=noise):
                got = self.svc.expectation(circuit, ["ZZ", "XI"], values={"t": 0.9}, shots=50, seed=3, noise=noise)
                self.assertEqual(got, baseline)

    def test_noisy_sampling_is_deterministic_and_consistent(self):
        bell = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        noise = {"two_qubit_depolarizing": 0.25}
        first = self.svc.expectation(bell, ["ZZ"], shots=400, seed=5, noise=noise)
        second = self.svc.expectation(bell, ["ZZ"], shots=400, seed=5, noise=noise)
        self.assertEqual(first, second)
        counts = first["results"][0]["counts"]
        self.assertEqual(counts["positive"] + counts["negative"], 400)
        # 含噪精确值 0.75，采样期望应落在附近。
        self.assertAlmostEqual(first["results"][0]["expectation"], 0.75, delta=0.15)


class NoisyGradientTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_noisy_gradient_matches_analytic(self):
        p = 0.3
        theta = 0.7
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        item = self.svc.gradient(circuit, ["Z"], values={"t": theta},
                                 noise={"single_qubit_depolarizing": p})["results"][0]
        self.assertAlmostEqual(item["expectation"], (1 - p) * math.cos(theta), places=12)
        self.assertAlmostEqual(item["gradients"]["t"], -(1 - p) * math.sin(theta), places=12)

    def test_noisy_gradient_matches_finite_difference(self):
        circuit = {"qubit_count": 2, "parameters": ["a", "b"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "a", "coefficient": 2.0, "offset": 0.3}},
            {"gate": "h", "target": 1},
            {"gate": "rz", "target": 1,
             "angle": {"parameter": "b", "coefficient": -0.5}},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rx", "target": 1, "angle": {"parameter": "a"}},
        ]}
        noise = {"single_qubit_depolarizing": 0.05, "two_qubit_depolarizing": 0.1}
        observables = ["ZZ", "XI", "IY", "YX"]
        values = {"a": 0.4, "b": 1.1}
        result = self.svc.gradient(circuit, observables, values=values, noise=noise)
        eps = 1e-6
        for k, item in enumerate(result["results"]):
            for name in ("a", "b"):
                plus = self.svc.expectation(
                    circuit, observables, values={**values, name: values[name] + eps}, noise=noise,
                )["results"][k]["expectation"]
                minus = self.svc.expectation(
                    circuit, observables, values={**values, name: values[name] - eps}, noise=noise,
                )["results"][k]["expectation"]
                self.assertAlmostEqual(item["gradients"][name], (plus - minus) / (2 * eps), places=5)

    def test_zero_noise_gradient_matches_baseline(self):
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        baseline = self.svc.gradient(circuit, ["ZZ", "XI"], values={"t": 0.9})
        got = self.svc.gradient(circuit, ["ZZ", "XI"], values={"t": 0.9},
                                noise={"two_qubit_depolarizing": 0.0})
        self.assertEqual(got, baseline)

    def test_unused_parameter_keeps_zero_gradient_with_noise(self):
        circuit = {"qubit_count": 1, "parameters": ["used", "unused"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "used"}},
        ]}
        result = self.svc.gradient(circuit, ["Z"], values={"used": 0.5, "unused": 1.25},
                                   noise={"single_qubit_depolarizing": 0.2})
        gradients = result["results"][0]["gradients"]
        self.assertEqual(list(gradients), ["used", "unused"])
        self.assertEqual(gradients["unused"], 0.0)


class NoiseValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def check_both_entries(self, noise, code, path):
        for entry in (self.svc.expectation, self.svc.gradient):
            with self.subTest(entry=entry.__name__, noise=noise):
                e = sim_err(entry, {"qubit_count": 1}, ["Z"], noise=noise)
                self.assertEqual((e.code, e.path), (code, path))

    def test_noise_must_be_object(self):
        for bad in (0.5, "x", [], True, [("single_qubit_depolarizing", 0.1)]):
            self.check_both_entries(bad, "invalid_noise_model", "noise")

    def test_probability_type_and_range(self):
        for bad in (True, "0.5", -0.1, 1.1, float("nan"), float("inf"), None, []):
            self.check_both_entries(
                {"single_qubit_depolarizing": bad},
                "invalid_noise_model", "noise.single_qubit_depolarizing",
            )
            self.check_both_entries(
                {"two_qubit_depolarizing": bad},
                "invalid_noise_model", "noise.two_qubit_depolarizing",
            )

    def test_boundary_probabilities_are_accepted(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        for p in (0, 0.0, 1, 1.0):
            with self.subTest(p=p):
                result = self.svc.expectation(circuit, ["Z"], noise={"single_qubit_depolarizing": p})
                self.assertIn(result["results"][0]["expectation"], (-1.0, 0.0))

    def test_unknown_fields_point_to_lexicographically_first(self):
        self.check_both_entries({"bogus": 1}, "invalid_noise_model", "noise.bogus")
        self.check_both_entries({"zzz": 0.1, "aaa": 0.2}, "invalid_noise_model", "noise.aaa")

    def test_single_validated_before_two_and_probabilities_before_unknown(self):
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"],
                    noise={"two_qubit_depolarizing": -1.0, "single_qubit_depolarizing": "bad"})
        self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise.single_qubit_depolarizing"))
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"],
                    noise={"single_qubit_depolarizing": -1.0, "unknown_field": 0.5})
        self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise.single_qubit_depolarizing"))

    def test_existing_validation_precedes_noise(self):
        # 电路、绑定、observable 校验先于 noise 校验。
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["ZZ"], noise="bad")
        self.assertEqual((e.code, e.path), ("invalid_observable", "observables[0]"))
        with self.assertRaises(CircuitValidationError):
            self.svc.expectation({"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]},
                                 ["Z"], noise="bad")

    def test_noisy_qubit_limit_is_ten(self):
        for entry in (self.svc.expectation, self.svc.gradient):
            with self.subTest(entry=entry.__name__):
                e = sim_err(entry, {"qubit_count": 11}, ["I" * 11],
                            noise={"single_qubit_depolarizing": 0.1})
                self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))
        # 10 个量子位在含噪时仍可用。
        result = self.svc.expectation({"qubit_count": 10}, ["I" * 10],
                                      noise={"single_qubit_depolarizing": 0.1})
        self.assertEqual(result["results"][0]["expectation"], 1.0)
        # 概率全零时保持 20 量子位上限。
        result = self.svc.expectation({"qubit_count": 11}, ["I" * 11],
                                      noise={"single_qubit_depolarizing": 0.0})
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_inputs_not_mutated_and_result_json_serializable(self):
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rz", "target": 1, "angle": {"parameter": "t"}},
        ]}
        noise = {"single_qubit_depolarizing": 0.1, "two_qubit_depolarizing": 0.2}
        snapshot = (copy.deepcopy(circuit), copy.deepcopy(noise))
        exact = self.svc.expectation(circuit, ["ZZ"], values={"t": 0.3}, noise=noise)
        sampled = self.svc.expectation(circuit, ["ZZ"], values={"t": 0.3}, shots=10, seed=1, noise=noise)
        gradient = self.svc.gradient(circuit, ["ZZ"], values={"t": 0.3}, noise=noise)
        self.assertEqual((circuit, noise), snapshot)
        for payload in (exact, sampled, gradient):
            text = json.dumps(payload, sort_keys=True)
            self.assertNotIn("-0.0", text)


if __name__ == "__main__":
    unittest.main()
