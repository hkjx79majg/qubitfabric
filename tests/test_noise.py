import copy
import json
import math
import unittest

from qubitfabric.service import Service, SimulationError


def sim_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except SimulationError as exc:
        return exc
    raise AssertionError("SimulationError not raised")


SINGLE = {"single_qubit_depolarizing": 0.5}
TWO = {"two_qubit_depolarizing": 0.25}


class NoisyExactExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def expectations(self, circuit, observables, noise, values=None):
        result = self.svc.expectation(circuit, observables, values=values, noise=noise)
        return [item["expectation"] for item in result["results"]]

    def test_x_then_depolarizing_scales_z(self):
        # D(ρ) = (1-p)ρ + p I/2：|1⟩ 上 ⟨Z⟩ 从 -1 变为 -(1-p)。
        circuit = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        values = self.expectations(circuit, ["Z", "I"], {"single_qubit_depolarizing": 0.5})
        self.assertAlmostEqual(values[0], -0.5, places=12)
        self.assertAlmostEqual(values[1], 1.0, places=12)

    def test_full_depolarizing_wipes_non_identity(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        values = self.expectations(circuit, ["X", "Z", "I"], {"single_qubit_depolarizing": 1.0})
        self.assertAlmostEqual(values[0], 0.0, places=12)
        self.assertAlmostEqual(values[1], 0.0, places=12)
        self.assertAlmostEqual(values[2], 1.0, places=12)

    def test_noise_applies_after_every_single_qubit_gate(self):
        # 两个 h 各跟一次 p=0.5 通道：净效果与一次通道不同。
        circuit = {"qubit_count": 1, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "h", "target": 0},
        ]}
        values = self.expectations(circuit, ["Z"], {"single_qubit_depolarizing": 0.5})
        # ρ1 = (|+⟩⟨+| + I/2)/2，h 后 ⟨Z⟩ 项被洗掉一半，再通道一次：
        # ⟨Z⟩ = (1/2)*(1/2)*1 = 0.25。
        self.assertAlmostEqual(values[0], 0.25, places=12)

    def test_rx_rotation_damped(self):
        theta = 0.4
        circuit = {"qubit_count": 1, "operations": [{"gate": "rx", "target": 0, "angle": theta}]}
        values = self.expectations(circuit, ["Z"], {"single_qubit_depolarizing": 0.3})
        self.assertAlmostEqual(values[0], 0.7 * math.cos(theta), places=12)

    def test_cx_two_qubit_channel_on_bell_state(self):
        circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        values = self.expectations(circuit, ["ZZ", "XX", "YY", "II"], {"two_qubit_depolarizing": 0.25})
        self.assertAlmostEqual(values[0], 0.75, places=12)
        self.assertAlmostEqual(values[1], 0.75, places=12)
        self.assertAlmostEqual(values[2], -0.75, places=12)
        self.assertAlmostEqual(values[3], 1.0, places=12)

    def test_single_channel_does_not_touch_cx(self):
        # 只给单比特通道时 cx 本身不引入噪声。
        circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        values = self.expectations(circuit, ["ZZ"], {"single_qubit_depolarizing": 0.0,
                                                     "two_qubit_depolarizing": 0.4})
        self.assertAlmostEqual(values[0], 0.6, places=12)

    def test_channel_order_follows_ir(self):
        # x 后 cx：双比特通道作用在翻转之后，⟨ZI⟩ = -(1-p)。
        circuit = {"qubit_count": 2, "operations": [
            {"gate": "x", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        values = self.expectations(circuit, ["ZI", "IZ"], {"two_qubit_depolarizing": 0.5})
        self.assertAlmostEqual(values[0], -0.5, places=12)
        self.assertAlmostEqual(values[1], -0.5, places=12)

    def test_zero_probabilities_match_noiseless_exactly(self):
        circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rz", "target": 1, "angle": 0.3},
        ]}
        observables = ["ZZ", "XX", "YY", "ZI"]
        baseline = self.svc.expectation(circuit, observables)
        for noise in (None, {},
                      {"single_qubit_depolarizing": 0.0},
                      {"two_qubit_depolarizing": 0.0},
                      {"single_qubit_depolarizing": 0.0, "two_qubit_depolarizing": 0.0}):
            with self.subTest(noise=noise):
                self.assertEqual(self.svc.expectation(circuit, observables, noise=noise), baseline)

    def test_parameterized_circuit_with_noise(self):
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        values = self.expectations(circuit, ["Z"], SINGLE, values={"t": math.pi})
        self.assertAlmostEqual(values[0], -0.5, places=12)


class NoisySampledExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}

    def test_counts_sum_to_shots_and_reproduce(self):
        args = (self.circuit, ["ZZ", "XX"])
        kwargs = {"shots": 300, "seed": 9, "noise": TWO}
        first = self.svc.expectation(*args, **kwargs)
        second = self.svc.expectation(*args, **kwargs)
        self.assertEqual(first, second)
        for item in first["results"]:
            counts = item["counts"]
            self.assertEqual(counts["positive"] + counts["negative"], 300)
            expected = (counts["positive"] - counts["negative"]) / 300
            self.assertEqual(item["expectation"], expected if expected != 0 else 0.0)

    def test_sampling_tracks_noisy_exact_value(self):
        # p=0.5 双比特通道后 ⟨ZZ⟩ = 0.5。
        result = self.svc.expectation(self.circuit, ["ZZ"], shots=20000, seed=4,
                                      noise={"two_qubit_depolarizing": 0.5})
        self.assertAlmostEqual(result["results"][0]["expectation"], 0.5, delta=0.05)

    def test_zero_noise_sampling_matches_noiseless(self):
        args = (self.circuit, ["ZZ", "XX"])
        baseline = self.svc.expectation(*args, shots=200, seed=6)
        noisy = self.svc.expectation(*args, shots=200, seed=6,
                                     noise={"single_qubit_depolarizing": 0.0,
                                            "two_qubit_depolarizing": 0.0})
        self.assertEqual(noisy, baseline)


class NoisyGradientTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_rx_gradient_damped_by_channel(self):
        theta = 0.3
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        item = self.svc.gradient(circuit, ["Z"], values={"t": theta},
                                 noise={"single_qubit_depolarizing": 0.2})["results"][0]
        self.assertAlmostEqual(item["expectation"], 0.8 * math.cos(theta), places=12)
        self.assertAlmostEqual(item["gradients"]["t"], -0.8 * math.sin(theta), places=12)

    def test_noisy_gradient_matches_finite_difference(self):
        circuit = {"qubit_count": 2, "parameters": ["a", "b"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "a", "coefficient": 2.0, "offset": 0.3}},
            {"gate": "h", "target": 1},
            {"gate": "rz", "target": 1, "angle": {"parameter": "b", "coefficient": -0.5}},
            {"gate": "cx", "control": 0, "target": 1},
            {"gate": "rx", "target": 1, "angle": {"parameter": "a"}},
        ]}
        observables = ["ZZ", "XI", "IY"]
        noise = {"single_qubit_depolarizing": 0.05, "two_qubit_depolarizing": 0.1}
        values = {"a": 0.4, "b": 1.1}
        result = self.svc.gradient(circuit, observables, values=values, noise=noise)
        eps = 1e-6
        for k, item in enumerate(result["results"]):
            for name in ("a", "b"):
                plus = dict(values, **{name: values[name] + eps})
                minus = dict(values, **{name: values[name] - eps})
                e_plus = self.svc.expectation(circuit, observables, values=plus, noise=noise)["results"][k]["expectation"]
                e_minus = self.svc.expectation(circuit, observables, values=minus, noise=noise)["results"][k]["expectation"]
                self.assertAlmostEqual(item["gradients"][name], (e_plus - e_minus) / (2 * eps), places=5)

    def test_noisy_base_expectation_matches_noisy_expectation_entry(self):
        circuit = {"qubit_count": 2, "parameters": ["t"], "operations": [
            {"gate": "h", "target": 0},
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        observables = ["ZZ", "XX"]
        kwargs = {"values": {"t": 0.8}, "noise": TWO}
        gradient = self.svc.gradient(circuit, observables, **kwargs)
        exact = self.svc.expectation(circuit, observables, **kwargs)
        for got, want in zip(gradient["results"], exact["results"]):
            self.assertEqual(got["expectation"], want["expectation"])

    def test_unused_parameter_and_structure_unchanged(self):
        circuit = {"qubit_count": 1, "parameters": ["used", "unused"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "used"}},
        ]}
        result = self.svc.gradient(circuit, ["Z"], values={"used": 0.5, "unused": 1.0}, noise=SINGLE)
        self.assertEqual(result["parameters"], ["used", "unused"])
        gradients = result["results"][0]["gradients"]
        self.assertEqual(list(gradients), ["used", "unused"])
        self.assertEqual(gradients["unused"], 0.0)

    def test_zero_noise_gradient_matches_noiseless(self):
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        baseline = self.svc.gradient(circuit, ["Z"], values={"t": 0.7})
        for noise in (None, {}, {"single_qubit_depolarizing": 0.0, "two_qubit_depolarizing": 0.0}):
            with self.subTest(noise=noise):
                self.assertEqual(self.svc.gradient(circuit, ["Z"], values={"t": 0.7}, noise=noise), baseline)


class NoiseValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_noise_must_be_object(self):
        for bad in (0, 1.5, "noise", [0.1], True, ()):
            with self.subTest(bad=bad):
                e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], noise=bad)
                self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise"))

    def test_non_string_field_name(self):
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], noise={1: 0.5})
        self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise"))

    def test_invalid_probabilities(self):
        cases = [
            ({"single_qubit_depolarizing": True}, "noise.single_qubit_depolarizing"),
            ({"single_qubit_depolarizing": "0.5"}, "noise.single_qubit_depolarizing"),
            ({"single_qubit_depolarizing": None}, "noise.single_qubit_depolarizing"),
            ({"single_qubit_depolarizing": -0.1}, "noise.single_qubit_depolarizing"),
            ({"single_qubit_depolarizing": 1.1}, "noise.single_qubit_depolarizing"),
            ({"single_qubit_depolarizing": math.nan}, "noise.single_qubit_depolarizing"),
            ({"single_qubit_depolarizing": math.inf}, "noise.single_qubit_depolarizing"),
            ({"two_qubit_depolarizing": 2.0}, "noise.two_qubit_depolarizing"),
        ]
        for noise, path in cases:
            with self.subTest(noise=noise):
                e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], noise=noise)
                self.assertEqual((e.code, e.path), ("invalid_noise_model", path))

    def test_single_validated_before_two(self):
        noise = {"single_qubit_depolarizing": 2.0, "two_qubit_depolarizing": 2.0}
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], noise=noise)
        self.assertEqual(e.path, "noise.single_qubit_depolarizing")

    def test_probabilities_validated_before_unknown_fields(self):
        noise = {"two_qubit_depolarizing": 2.0, "unknown": 0.1}
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], noise=noise)
        self.assertEqual(e.path, "noise.two_qubit_depolarizing")

    def test_unknown_field_reports_lexicographically_smallest(self):
        noise = {"zzz": 0.1, "aaa": 0.2, "single_qubit_depolarizing": 0.1}
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], noise=noise)
        self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise.aaa"))

    def test_boundary_probabilities_accepted(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        result = self.svc.expectation(circuit, ["Z"], noise={
            "single_qubit_depolarizing": 0,
            "two_qubit_depolarizing": 1,
        })
        self.assertEqual(result["results"][0]["expectation"], -1.0)

    def test_gradient_validates_noise(self):
        e = sim_err(self.svc.gradient, {"qubit_count": 1}, ["Z"], noise="x")
        self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise"))
        e = sim_err(self.svc.gradient, {"qubit_count": 1}, ["Z"],
                    noise={"two_qubit_depolarizing": 1.5})
        self.assertEqual((e.code, e.path), ("invalid_noise_model", "noise.two_qubit_depolarizing"))

    def test_existing_validation_precedes_noise(self):
        from qubitfabric.service import CircuitValidationError
        with self.assertRaises(CircuitValidationError):
            self.svc.expectation({"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]},
                                 ["Z"], noise=5)
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, [], noise=5)
        self.assertEqual(e.code, "invalid_observables")
        e = sim_err(self.svc.expectation, {"qubit_count": 1}, ["Z"], shots=0, noise=5)
        self.assertEqual(e.code, "invalid_shots")

    def test_noisy_qubit_limit_is_ten(self):
        for noise in ({"single_qubit_depolarizing": 0.1}, {"two_qubit_depolarizing": 0.1}):
            with self.subTest(noise=noise):
                e = sim_err(self.svc.expectation, {"qubit_count": 11}, ["I" * 11], noise=noise)
                self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))
                e = sim_err(self.svc.gradient, {"qubit_count": 11}, ["I" * 11], noise=noise)
                self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))
        # 边界：10 个量子位含噪可用。
        result = self.svc.expectation({"qubit_count": 10}, ["I" * 10], noise=SINGLE)
        self.assertEqual(result["results"][0]["expectation"], 1.0)

    def test_zero_noise_keeps_twenty_qubit_limit(self):
        # 全零概率等价于无噪声：11 个量子位仍然可用，21 个才超限。
        result = self.svc.expectation({"qubit_count": 11}, ["I" * 11],
                                      noise={"single_qubit_depolarizing": 0.0})
        self.assertEqual(result["results"][0]["expectation"], 1.0)
        e = sim_err(self.svc.expectation, {"qubit_count": 21}, ["I" * 21],
                    noise={"single_qubit_depolarizing": 0.0})
        self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))

    def test_noise_input_not_mutated_and_output_json_native(self):
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t"}},
        ]}
        noise = {"single_qubit_depolarizing": 0.1, "two_qubit_depolarizing": 0.2}
        snapshot = (copy.deepcopy(circuit), copy.deepcopy(noise))
        result = self.svc.expectation(circuit, ["Z"], values={"t": 0.5}, shots=10, seed=1, noise=noise)
        gradient = self.svc.gradient(circuit, ["Z"], values={"t": 0.5}, noise=noise)
        self.assertEqual((circuit, noise), snapshot)
        json.dumps(result, sort_keys=True)
        payload = json.dumps(gradient, sort_keys=True)
        self.assertNotIn("-0.0", payload)


if __name__ == "__main__":
    unittest.main()
