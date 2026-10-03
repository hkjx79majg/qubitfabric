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


RX_CIRCUIT = {"qubit_count": 1, "parameters": ["theta"],
              "operations": [{"gate": "rx", "target": 0, "angle": {"parameter": "theta"}}]}
Z_TERM = [{"observable": "Z", "coefficient": 1.0}]
BASE_CONFIG = {"method": "gradient_descent", "learning_rate": 1.0,
               "max_iterations": 1, "tolerance": 0.0}


class GradientDescentTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def optimize(self, circuit, terms, values, config, **kwargs):
        return self.svc.optimize(circuit, terms, values=values, config=config, **kwargs)

    def test_rx_z_matches_analytic_trajectory(self):
        # 最小化 cos(θ)，梯度 -sin(θ)：θ ← θ + lr sin(θ)。
        config = {"method": "gradient_descent", "learning_rate": 0.1,
                  "max_iterations": 5, "tolerance": 0.0}
        result = self.optimize(RX_CIRCUIT, Z_TERM, {"theta": 0.3}, config)
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 5)
        self.assertEqual(result["parameters"], ["theta"])
        self.assertEqual([h["iteration"] for h in result["history"]], [0, 1, 2, 3, 4, 5])

        trajectory = [0.3]
        for _ in range(5):
            current = trajectory[-1]
            trajectory.append(current + 0.1 * math.sin(current))
        for h, theta in zip(result["history"], trajectory):
            self.assertAlmostEqual(h["values"]["theta"], theta, places=12)
            self.assertAlmostEqual(h["objective"], math.cos(theta), places=12)
            self.assertAlmostEqual(h["gradient_norm"], abs(math.sin(theta)), places=12)
        self.assertAlmostEqual(result["final_values"]["theta"], trajectory[-1], places=12)
        self.assertAlmostEqual(result["final_objective"], result["history"][-1]["objective"])

    def test_weighted_terms_accumulate_in_input_order(self):
        # h; rz(t) 之后 <X>=cos(t)、<Y>=sin(t)；H = X - 0.5 Y。
        circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "h", "target": 0},
            {"gate": "rz", "target": 0, "angle": {"parameter": "t"}},
        ]}
        terms = [{"observable": "X", "coefficient": 1.0},
                 {"observable": "Y", "coefficient": -0.5}]
        t0 = 0.7
        result = self.optimize(
            circuit, terms, {"t": t0},
            {"method": "gradient_descent", "learning_rate": 0.2,
             "max_iterations": 1, "tolerance": 0.0},
        )
        objective = math.cos(t0) - 0.5 * math.sin(t0)
        gradient = -math.sin(t0) - 0.5 * math.cos(t0)
        self.assertAlmostEqual(result["history"][0]["objective"], objective, places=12)
        self.assertAlmostEqual(result["history"][0]["gradient_norm"], abs(gradient), places=12)
        self.assertAlmostEqual(result["history"][1]["values"]["t"], t0 - 0.2 * gradient, places=12)

    def test_multiple_parameters_update_simultaneously_in_declaration_order(self):
        circuit = {"qubit_count": 1, "parameters": ["a", "b"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "a"}},
            {"gate": "rx", "target": 0, "angle": {"parameter": "b"}},
        ]}
        result = self.optimize(
            circuit, Z_TERM, {"a": 0.2, "b": 0.5},
            {"method": "gradient_descent", "learning_rate": 0.1,
             "max_iterations": 1, "tolerance": 0.0},
        )
        self.assertEqual(list(result["history"][0]["values"]), ["a", "b"])
        gradient = -math.sin(0.7)
        self.assertAlmostEqual(result["history"][1]["values"]["a"], 0.2 - 0.1 * gradient, places=12)
        self.assertAlmostEqual(result["history"][1]["values"]["b"], 0.5 - 0.1 * gradient, places=12)
        self.assertAlmostEqual(result["history"][0]["gradient_norm"],
                               math.sqrt(2.0) * abs(gradient), places=12)

    def test_tolerance_stops_at_initial_point(self):
        result = self.optimize(
            RX_CIRCUIT, Z_TERM, {"theta": 0.0},
            {"method": "gradient_descent", "learning_rate": 0.1,
             "max_iterations": 3, "tolerance": 0.0},
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(len(result["history"]), 1)

    def test_norm_equal_to_tolerance_converges(self):
        gradient = self.svc.gradient(RX_CIRCUIT, ["Z"], values={"theta": 0.3})
        norm = abs(gradient["results"][0]["gradients"]["theta"])
        result = self.optimize(
            RX_CIRCUIT, Z_TERM, {"theta": 0.3},
            {"method": "gradient_descent", "learning_rate": 0.1,
             "max_iterations": 5, "tolerance": norm},
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)

    def test_max_iterations_without_convergence(self):
        result = self.optimize(
            RX_CIRCUIT, Z_TERM, {"theta": 0.3},
            {"method": "gradient_descent", "learning_rate": 0.1,
             "max_iterations": 4, "tolerance": 0.0},
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 4)
        self.assertEqual(len(result["history"]), 5)

    def test_zero_parameter_circuit_converges_without_updates(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        result = self.optimize(
            circuit, [{"observable": "X", "coefficient": 2.0}], {},
            {"method": "adam", "learning_rate": 1.0, "max_iterations": 3, "tolerance": 0.0},
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["parameters"], [])
        self.assertEqual(result["final_values"], {})
        self.assertEqual(result["history"][0]["values"], {})
        self.assertEqual(result["history"][0]["gradient_norm"], 0.0)
        self.assertAlmostEqual(result["final_objective"], 2.0, places=12)


class AdamTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_first_step_matches_reference_formula(self):
        theta0 = 0.4
        beta1, beta2, epsilon, lr = 0.9, 0.999, 1e-8, 0.05
        g0 = -math.sin(theta0)
        m_hat = (1 - beta1) * g0 / (1 - beta1)
        v_hat = (1 - beta2) * g0 * g0 / (1 - beta2)
        expected = theta0 - lr * m_hat / (math.sqrt(v_hat) + epsilon)

        result = self.svc.optimize(
            RX_CIRCUIT, Z_TERM, {"theta": theta0},
            {"method": "adam", "learning_rate": lr, "max_iterations": 1, "tolerance": 0.0},
        )
        self.assertAlmostEqual(result["history"][1]["values"]["theta"], expected, places=12)

    def test_custom_betas_and_epsilon_are_used(self):
        common = {"method": "adam", "learning_rate": 0.05, "max_iterations": 1, "tolerance": 0.0}
        defaults = self.svc.optimize(RX_CIRCUIT, Z_TERM, {"theta": 0.4}, common)
        custom = self.svc.optimize(
            RX_CIRCUIT, Z_TERM, {"theta": 0.4},
            {**common, "beta1": 0.8, "beta2": 0.95, "epsilon": 1e-7},
        )
        self.assertNotEqual(
            custom["history"][1]["values"]["theta"],
            defaults["history"][1]["values"]["theta"],
        )

    def test_descends_to_minimum(self):
        circuit = {"qubit_count": 1, "parameters": ["a"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "a", "coefficient": 2.0}},
        ]}
        result = self.svc.optimize(
            circuit, Z_TERM, {"a": 0.3},
            {"method": "adam", "learning_rate": 0.05, "max_iterations": 50, "tolerance": 0.0},
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 50)
        self.assertLess(result["final_objective"], -0.99)

    def test_zero_parameter_circuit_converges(self):
        result = self.svc.optimize(
            {"qubit_count": 0}, [{"observable": "", "coefficient": 1.0}], {},
            {"method": "adam", "learning_rate": 1.0, "max_iterations": 2, "tolerance": 0.0},
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["final_objective"], 1.0)


class NoisyOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_noise_is_applied_and_deterministic(self):
        config = {"method": "gradient_descent", "learning_rate": 0.1,
                  "max_iterations": 3, "tolerance": 0.0}
        noise = {"single_qubit_depolarizing": 0.3}
        first = self.svc.optimize(RX_CIRCUIT, Z_TERM, {"theta": 0.7}, config, noise=noise)
        second = self.svc.optimize(RX_CIRCUIT, Z_TERM, {"theta": 0.7}, config, noise=noise)
        clean = self.svc.optimize(RX_CIRCUIT, Z_TERM, {"theta": 0.7}, config)
        self.assertEqual(first, second)
        self.assertNotEqual(first["history"][0]["objective"], clean["history"][0]["objective"])
        # 含噪目标：(1-p) cos(θ)。
        self.assertAlmostEqual(first["history"][0]["objective"], 0.7 * math.cos(0.7), places=12)

    def test_noisy_qubit_limit_is_ten(self):
        with self.assertRaises(SimulationError) as ctx:
            self.svc.optimize(
                {"qubit_count": 11}, [{"observable": "I" * 11, "coefficient": 1.0}], {},
                BASE_CONFIG, noise={"single_qubit_depolarizing": 0.1},
            )
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("state_space_too_large", "qubit_count"))


class DeterminismTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_repeated_calls_identical_and_inputs_untouched(self):
        config = {"method": "adam", "learning_rate": 0.1, "max_iterations": 4, "tolerance": 0.0}
        values = {"theta": 0.3}
        snapshot = copy.deepcopy((RX_CIRCUIT, Z_TERM, values, config))
        first = self.svc.optimize(RX_CIRCUIT, Z_TERM, values=values, config=config)
        second = self.svc.optimize(RX_CIRCUIT, Z_TERM, values=values, config=config)
        self.assertEqual(first, second)
        self.assertEqual((RX_CIRCUIT, Z_TERM, values, config), snapshot)
        payload = json.dumps(first, sort_keys=True)
        self.assertNotIn("-0.0", payload)

    def test_initial_point_is_first_history_entry(self):
        result = self.svc.optimize(
            RX_CIRCUIT, Z_TERM, {"theta": 0.3},
            {"method": "gradient_descent", "learning_rate": 0.1,
             "max_iterations": 2, "tolerance": 0.0},
        )
        self.assertEqual(result["history"][0]["iteration"], 0)
        self.assertEqual(result["history"][0]["values"], {"theta": 0.3})
        self.assertEqual(result["final_values"], result["history"][-1]["values"])


class TermsValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_terms_must_be_non_empty_array(self):
        for bad in (None, [], "z", {}, 1):
            with self.subTest(bad=bad):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, bad, config=BASE_CONFIG)
                self.assertEqual((e.code, e.path), ("invalid_terms", "terms"))

    def test_invalid_term_locations(self):
        cases = [
            ([{"observable": "Z"}], "terms[0].coefficient"),
            (["not-an-object"], "terms[0]"),
            ([{"observable": "Z", "coefficient": math.inf}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": math.nan}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": "1"}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": True}], "terms[0].coefficient"),
            ([{"observable": "Z", "coefficient": None}], "terms[0].coefficient"),
            ([{"coefficient": 1.0}], "terms[0].observable"),
            ([{"observable": "Z", "coefficient": 1.0, "bogus": 2}], "terms[0].bogus"),
            ([{"observable": "Z", "coefficient": 1.0}, {"observable": "Z"}],
             "terms[1].coefficient"),
        ]
        for terms, path in cases:
            with self.subTest(path=path):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, terms, config=BASE_CONFIG)
                self.assertEqual((e.code, e.path), ("invalid_term", path))

    def test_unknown_term_field_reported_lexicographically_first(self):
        terms = [{"observable": "Z", "coefficient": 1.0, "zzz": 1, "aaa": 2}]
        e = opt_err(self.svc.optimize, {"qubit_count": 1}, terms, config=BASE_CONFIG)
        self.assertEqual((e.code, e.path), ("invalid_term", "terms[0].aaa"))

    def test_observable_mismatch_keeps_simulation_error(self):
        for bad in ("ZZ", "z", 1):
            with self.subTest(bad=bad):
                with self.assertRaises(SimulationError) as ctx:
                    self.svc.optimize(
                        {"qubit_count": 1},
                        [{"observable": bad, "coefficient": 1.0}],
                        config=BASE_CONFIG,
                    )
                self.assertEqual((ctx.exception.code, ctx.exception.path),
                                 ("invalid_observable", "terms[0].observable"))


class ConfigValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_config_must_be_object(self):
        for bad in (None, [], "x", 1, True):
            with self.subTest(bad=bad):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM, config=bad)
                self.assertEqual((e.code, e.path), ("invalid_optimizer_config", "config"))

    def test_unknown_method(self):
        config = {"learning_rate": 1.0, "max_iterations": 1, "tolerance": 0.0}
        e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM, config=config)
        self.assertEqual((e.code, e.path), ("unsupported_optimizer", "config.method"))
        e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM, config={**config, "method": "sgd"})
        self.assertEqual((e.code, e.path), ("unsupported_optimizer", "config.method"))
        e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM, config={**config, "method": 1})
        self.assertEqual((e.code, e.path), ("unsupported_optimizer", "config.method"))

    def test_unknown_options_reported_lexicographically_first(self):
        e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM,
                    config={**BASE_CONFIG, "zzz": 1, "aaa": 2})
        self.assertEqual((e.code, e.path), ("unknown_optimizer_option", "config.aaa"))

    def test_adam_only_options_unknown_for_gradient_descent(self):
        for name, value in (("beta1", 0.9), ("beta2", 0.999), ("epsilon", 1e-8)):
            with self.subTest(name=name):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM,
                            config={**BASE_CONFIG, name: value})
                self.assertEqual((e.code, e.path),
                                 ("unknown_optimizer_option", f"config.{name}"))

    def test_invalid_learning_rate(self):
        for bad in (0, -1.0, math.inf, math.nan, "1", True, None):
            with self.subTest(bad=bad):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM,
                            config={**BASE_CONFIG, "learning_rate": bad})
                self.assertEqual((e.code, e.path),
                                 ("invalid_optimizer_option", "config.learning_rate"))

    def test_invalid_max_iterations(self):
        for bad in (0, -1, 1.5, "1", True, None):
            with self.subTest(bad=bad):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM,
                            config={**BASE_CONFIG, "max_iterations": bad})
                self.assertEqual((e.code, e.path),
                                 ("invalid_optimizer_option", "config.max_iterations"))

    def test_invalid_tolerance(self):
        for bad in (-0.1, math.inf, math.nan, "0", True, None):
            with self.subTest(bad=bad):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM,
                            config={**BASE_CONFIG, "tolerance": bad})
                self.assertEqual((e.code, e.path),
                                 ("invalid_optimizer_option", "config.tolerance"))

    def test_invalid_adam_betas_and_epsilon(self):
        adam_base = {"method": "adam", "learning_rate": 1.0, "max_iterations": 1, "tolerance": 0.0}
        for name, bad in (
            ("beta1", -0.1), ("beta1", 1.0), ("beta1", "x"), ("beta1", None),
            ("beta2", -0.1), ("beta2", 1.0), ("beta2", "x"),
            ("epsilon", 0.0), ("epsilon", -1e-9), ("epsilon", "x"),
        ):
            with self.subTest(name=name, bad=bad):
                e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM,
                            config={**adam_base, name: bad})
                self.assertEqual((e.code, e.path),
                                 ("invalid_optimizer_option", f"config.{name}"))

    def test_missing_required_options(self):
        for missing in ("learning_rate", "max_iterations", "tolerance"):
            config = {k: v for k, v in BASE_CONFIG.items() if k != missing}
            e = opt_err(self.svc.optimize, {"qubit_count": 1}, Z_TERM, config=config)
            self.assertEqual((e.code, e.path),
                             ("invalid_optimizer_option", f"config.{missing}"))


class ExistingValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_circuit_and_binding_errors_are_reused(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.optimize(
                {"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]},
                Z_TERM, values={}, config=BASE_CONFIG,
            )
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.optimize(RX_CIRCUIT, Z_TERM, config=BASE_CONFIG)
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("missing_parameter", "theta"))
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.optimize(RX_CIRCUIT, Z_TERM,
                              values={"theta": 1.0, "nope": 2.0}, config=BASE_CONFIG)
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("unknown_parameter", "nope"))

    def test_noise_and_state_space_errors_are_reused(self):
        with self.assertRaises(SimulationError) as ctx:
            self.svc.optimize({"qubit_count": 1}, Z_TERM, config=BASE_CONFIG, noise="bad")
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("invalid_noise_model", "noise"))
        with self.assertRaises(SimulationError) as ctx:
            self.svc.optimize(
                {"qubit_count": 21}, [{"observable": "I" * 21, "coefficient": 1.0}],
                config=BASE_CONFIG,
            )
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("state_space_too_large", "qubit_count"))

    def test_quantum_errors_precede_iteration_and_return_values(self):
        # 配置与 terms 合法，但绑定缺失：在任何更新前抛出。
        with self.assertRaises(ParameterBindingError):
            self.svc.optimize(RX_CIRCUIT, Z_TERM, config={
                "method": "gradient_descent", "learning_rate": 10.0,
                "max_iterations": 1, "tolerance": 0.0,
            })


if __name__ == "__main__":
    unittest.main()
