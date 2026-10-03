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


def opt_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except OptimizationError as exc:
        return exc
    raise AssertionError("OptimizationError not raised")


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


def rx_circuit():
    return {"qubit_count": 1, "parameters": ["theta"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
    ]}


def gd_config(**overrides):
    config = {"method": "gradient_descent", "learning_rate": 0.5,
              "max_iterations": 100, "tolerance": 1e-9}
    config.update(overrides)
    return config


class GradientDescentOptimizeTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_minimizes_cosine_objective(self):
        # 目标 cos(θ)，最小值 -1 在 θ=π。
        result = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": 0.1}, gd_config(),
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["parameters"], ["theta"])
        self.assertAlmostEqual(result["final_values"]["theta"], math.pi, places=6)
        self.assertAlmostEqual(result["final_objective"], -1.0, places=9)
        self.assertEqual(result["iterations"], len(result["history"]) - 1)
        self.assertEqual(
            [entry["iteration"] for entry in result["history"]],
            list(range(result["iterations"] + 1)),
        )

    def test_history_records_initial_point_and_updates(self):
        # 单参数 rx：目标 cos(θ)，梯度 -sin(θ)，学习率 0.5。
        theta = 0.3
        result = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": theta}, gd_config(max_iterations=5, tolerance=0.0),
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 5)
        expected = [theta]
        for _ in range(5):
            expected.append(expected[-1] + 0.5 * math.sin(expected[-1]))
        for entry, value in zip(result["history"], expected):
            self.assertAlmostEqual(entry["values"]["theta"], value, places=12)
            self.assertAlmostEqual(entry["objective"], math.cos(value), places=12)
            self.assertAlmostEqual(entry["gradient_norm"], abs(math.sin(value)), places=12)
        self.assertAlmostEqual(result["final_values"]["theta"], expected[-1], places=12)
        self.assertAlmostEqual(result["final_objective"], math.cos(expected[-1]), places=12)

    def test_weighted_multi_term_objective(self):
        # rx(θ)|0⟩：⟨Z⟩ = cos(θ)，⟨Y⟩ = -sin(θ)。
        # 目标 2*cos(θ) - 0.5*sin(θ)，梯度 -2*sin(θ) - 0.5*cos(θ)。
        terms = [
            {"observable": "Z", "coefficient": 2.0},
            {"observable": "Y", "coefficient": 0.5},
        ]
        theta = 0.4
        result = self.svc.optimize(
            rx_circuit(), terms, {"theta": theta},
            gd_config(max_iterations=1, tolerance=0.0),
        )
        grad = -2.0 * math.sin(theta) - 0.5 * math.cos(theta)
        new_theta = theta - 0.5 * grad
        self.assertEqual(result["iterations"], 1)
        self.assertAlmostEqual(result["history"][0]["objective"],
                               2.0 * math.cos(theta) - 0.5 * math.sin(theta), places=12)
        self.assertAlmostEqual(result["history"][0]["gradient_norm"], abs(grad), places=12)
        self.assertAlmostEqual(result["final_values"]["theta"], new_theta, places=12)

    def test_simultaneous_multi_parameter_update(self):
        # 目标 cos(a) + cos(b)：两参数按各自梯度同时更新。
        circuit = {"qubit_count": 2, "parameters": ["a", "b"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "a"}},
            {"gate": "rx", "target": 1, "angle": {"parameter": "b"}},
        ]}
        terms = [{"observable": "ZI", "coefficient": 1.0},
                 {"observable": "IZ", "coefficient": 1.0}]
        result = self.svc.optimize(circuit, terms, {"a": 0.2, "b": 0.9},
                                   gd_config(max_iterations=1, tolerance=0.0))
        values = result["final_values"]
        self.assertAlmostEqual(values["a"], 0.2 + 0.5 * math.sin(0.2), places=12)
        self.assertAlmostEqual(values["b"], 0.9 + 0.5 * math.sin(0.9), places=12)
        norm = math.hypot(math.sin(0.2), math.sin(0.9))
        self.assertAlmostEqual(result["history"][0]["gradient_norm"], norm, places=12)

    def test_zero_parameter_circuit_converges_without_updates(self):
        result = self.svc.optimize(
            {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]},
            [{"observable": "X", "coefficient": 1.0}], {}, gd_config(),
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["parameters"], [])
        self.assertEqual(result["final_values"], {})
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["history"][0]["gradient_norm"], 0.0)
        self.assertAlmostEqual(result["final_objective"], 1.0, places=12)

    def test_initial_point_already_converged(self):
        result = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": math.pi}, gd_config(tolerance=1e-6),
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(len(result["history"]), 1)

    def test_max_iterations_caps_updates(self):
        result = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": 0.1}, gd_config(max_iterations=3, tolerance=0.0),
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(len(result["history"]), 4)

    def test_noise_and_coefficient_order(self):
        noise = {"single_qubit_depolarizing": 0.1}
        noisy = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": 0.5}, gd_config(max_iterations=1, tolerance=0.0), noise=noise,
        )
        # 含噪期望值与梯度应与 svc.gradient 的含噪结果一致。
        grad = self.svc.gradient(rx_circuit(), ["Z"], values={"theta": 0.5}, noise=noise)
        expected = grad["results"][0]
        self.assertAlmostEqual(noisy["history"][0]["objective"],
                               expected["expectation"], places=12)
        self.assertAlmostEqual(noisy["history"][0]["gradient_norm"],
                               abs(expected["gradients"]["theta"]), places=12)

    def test_input_not_mutated_and_result_json_serializable(self):
        circuit = rx_circuit()
        terms = [{"observable": "Z", "coefficient": 1.0}]
        values = {"theta": 0.3}
        config = gd_config(max_iterations=2)
        snapshot = (copy.deepcopy(circuit), copy.deepcopy(terms),
                    dict(values), dict(config))
        result = self.svc.optimize(circuit, terms, values, config)
        self.assertEqual((circuit, terms, values, config), snapshot)
        payload = json.dumps(result, sort_keys=True)
        self.assertNotIn("-0.0", payload)


class AdamOptimizeTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_first_step_matches_bias_corrected_rule(self):
        # 第一步：m̂ = g，v̂ = g²，更新量 lr * g / (|g| + ε)。
        theta = 0.3
        epsilon = 1e-8
        result = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": theta},
            {"method": "adam", "learning_rate": 0.1,
             "max_iterations": 1, "tolerance": 0.0},
        )
        grad = -math.sin(theta)
        expected = theta - 0.1 * grad / (abs(grad) + epsilon)
        self.assertAlmostEqual(result["final_values"]["theta"], expected, places=12)

    def test_matches_reference_adam_recurrence(self):
        theta0 = 0.3
        lr, beta1, beta2, epsilon = 0.05, 0.8, 0.9, 1e-6
        result = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": theta0},
            {"method": "adam", "learning_rate": lr, "max_iterations": 10,
             "tolerance": 0.0, "beta1": beta1, "beta2": beta2, "epsilon": epsilon},
        )
        theta = theta0
        m = v = 0.0
        for t in range(1, 11):
            grad = -math.sin(theta)
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            theta -= lr * (m / (1.0 - beta1 ** t)) / (math.sqrt(v / (1.0 - beta2 ** t)) + epsilon)
            entry = result["history"][t]
            self.assertAlmostEqual(entry["values"]["theta"], theta, places=12)
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 10)

    def test_adam_converges_to_minimum(self):
        result = self.svc.optimize(
            rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
            {"theta": 0.2},
            {"method": "adam", "learning_rate": 0.2,
             "max_iterations": 500, "tolerance": 1e-7},
        )
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["final_objective"], -1.0, places=6)


class OptimizeValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.terms = [{"observable": "Z", "coefficient": 1.0}]
        self.values = {"theta": 0.3}

    def optimize(self, **overrides):
        args = {
            "circuit": self.circuit,
            "terms": self.terms,
            "values": self.values,
            "config": gd_config(),
        }
        args.update(overrides)
        return self.svc.optimize(args["circuit"], args["terms"], args["values"],
                                 args["config"], noise=args.get("noise"))

    def test_terms_must_be_non_empty_array(self):
        for bad in (None, "Z", {"0": "Z"}, [], 1):
            with self.subTest(bad=bad):
                e = opt_err(self.optimize, terms=bad)
                self.assertEqual((e.code, e.path), ("invalid_terms", "terms"))

    def test_invalid_term_structure(self):
        cases = [
            (["Z"], "terms[0]"),
            ([{"coefficient": 1.0}], "terms[0].observable"),
            ([{"observable": 1, "coefficient": 1.0}], "terms[0].observable"),
            ([{"observable": "Z"}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": "x"}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": True}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": math.inf}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": 1.0, "extra": 1}], "terms[0].extra"),
        ]
        for terms, path in cases:
            with self.subTest(terms=terms):
                e = opt_err(self.optimize, terms=terms)
                self.assertEqual((e.code, e.path), ("invalid_term", path))

    def test_first_invalid_term_is_reported(self):
        terms = [{"observable": "Z", "coefficient": 1.0}, "nope"]
        e = opt_err(self.optimize, terms=terms)
        self.assertEqual((e.code, e.path), ("invalid_term", "terms[1]"))

    def test_config_must_be_object(self):
        for bad in (None, [], "adam", 1):
            with self.subTest(bad=bad):
                e = opt_err(self.optimize, config=bad)
                self.assertEqual((e.code, e.path), ("invalid_optimizer_config", "config"))

    def test_unknown_or_missing_method(self):
        for config in (gd_config(method="nelder_mead"), gd_config(method=1),
                       {"learning_rate": 0.5, "max_iterations": 10, "tolerance": 0.0}):
            with self.subTest(config=config):
                e = opt_err(self.optimize, config=config)
                self.assertEqual((e.code, e.path), ("unsupported_optimizer", "config.method"))

    def test_unknown_option_lexicographic_first(self):
        e = opt_err(self.optimize, config=gd_config(zeta=1, alpha=2))
        self.assertEqual((e.code, e.path), ("unknown_optimizer_option", "config.alpha"))

    def test_invalid_options(self):
        cases = [
            (gd_config(learning_rate=0), "config.learning_rate"),
            (gd_config(learning_rate=-1), "config.learning_rate"),
            (gd_config(learning_rate=math.inf), "config.learning_rate"),
            (gd_config(learning_rate="0.5"), "config.learning_rate"),
            (gd_config(max_iterations=0), "config.max_iterations"),
            (gd_config(max_iterations=1.5), "config.max_iterations"),
            (gd_config(max_iterations=True), "config.max_iterations"),
            (gd_config(tolerance=-1e-9), "config.tolerance"),
            (gd_config(tolerance=math.nan), "config.tolerance"),
            ({"method": "adam", "learning_rate": 0.1, "max_iterations": 10,
              "tolerance": 0.0, "beta1": 1.0}, "config.beta1"),
            ({"method": "adam", "learning_rate": 0.1, "max_iterations": 10,
              "tolerance": 0.0, "beta2": -0.1}, "config.beta2"),
            ({"method": "adam", "learning_rate": 0.1, "max_iterations": 10,
              "tolerance": 0.0, "epsilon": 0.0}, "config.epsilon"),
        ]
        for config, path in cases:
            with self.subTest(config=config):
                e = opt_err(self.optimize, config=config)
                self.assertEqual((e.code, e.path), ("invalid_optimizer_option", path))

    def test_missing_required_options(self):
        for key in ("learning_rate", "max_iterations", "tolerance"):
            with self.subTest(missing=key):
                config = gd_config()
                del config[key]
                e = opt_err(self.optimize, config=config)
                self.assertEqual((e.code, e.path), ("invalid_optimizer_option", f"config.{key}"))

    def test_circuit_and_binding_errors_are_reused(self):
        with self.assertRaises(CircuitValidationError):
            self.optimize(circuit={"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]})
        e = bind_err(self.optimize, values={})
        self.assertEqual((e.code, e.path), ("missing_parameter", "theta"))
        e = bind_err(self.optimize, values={"theta": 0.3, "nope": 1})
        self.assertEqual((e.code, e.path), ("unknown_parameter", "nope"))

    def test_observable_and_noise_errors_are_reused(self):
        e = sim_err(self.optimize, terms=[{"observable": "ZZ", "coefficient": 1.0}])
        self.assertEqual((e.code, e.path), ("invalid_observable", "observables[0]"))
        e = sim_err(self.optimize, terms=[{"observable": "z", "coefficient": 1.0}])
        self.assertEqual((e.code, e.path), ("invalid_observable", "observables[0]"))
        e = sim_err(self.optimize, noise={"single_qubit_depolarizing": 2.0})
        self.assertEqual(e.code, "invalid_noise_model")

    def test_state_space_limit_is_reused(self):
        e = sim_err(
            self.svc.optimize,
            {"qubit_count": 21}, [{"observable": "I" * 21, "coefficient": 1.0}],
            {}, gd_config(),
        )
        self.assertEqual((e.code, e.path), ("state_space_too_large", "qubit_count"))

    def test_validation_precedes_iteration(self):
        # 非法配置与非法项都不产生任何历史；异常在迭代前抛出。
        for bad_call in (
            lambda: self.optimize(terms=[]),
            lambda: self.optimize(config=gd_config(learning_rate=-1)),
        ):
            with self.assertRaises(OptimizationError):
                bad_call()


if __name__ == "__main__":
    unittest.main()
