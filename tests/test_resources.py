import copy
import unittest

from qubitfabric.service import (
    CircuitValidationError,
    OptimizationError,
    ParameterBindingError,
    ResourceEstimationError,
    Service,
    SimulationError,
)


def res_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ResourceEstimationError as exc:
        return exc
    raise AssertionError("ResourceEstimationError not raised")


def variational_circuit():
    return {"qubit_count": 2, "parameters": ["theta"], "operations": [
        {"gate": "h", "target": 0},
        {"gate": "cx", "control": 0, "target": 1},
        {"gate": "rz", "target": 1, "angle": {"parameter": "theta", "coefficient": 2.0}},
    ]}


def exact_request(**overrides):
    request = {"type": "exact_expectation", "observables": ["ZZ"], "values": {"theta": 0.3}}
    request.update(overrides)
    return request


def gd_config(**overrides):
    config = {"method": "gradient_descent", "learning_rate": 0.5,
              "max_iterations": 5, "tolerance": 0.0}
    config.update(overrides)
    return config


def optimization_request(**overrides):
    request = {
        "type": "optimization",
        "terms": [{"observable": "ZZ", "coefficient": 1.0}],
        "values": {"theta": 0.3},
        "config": gd_config(),
    }
    request.update(overrides)
    return request


class EstimateResourcesMetricTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_exact_expectation_counts_single_evaluation(self):
        result = self.svc.estimate_resources(variational_circuit(), exact_request())
        self.assertEqual(result, {
            "qubit_count": 2,
            "type": "exact_expectation",
            "representation": "state_vector",
            "state_elements": 4,
            "state_bytes": 64,
            "circuit_evaluations": 1,
            "gate_applications": 3,
            "total_shots": 0,
            "runtime_supported": True,
            "admitted": True,
            "exceeded": [],
        })

    def test_sampled_expectation_counts_shots_per_observable(self):
        result = self.svc.estimate_resources(
            variational_circuit(),
            {"type": "sampled_expectation", "observables": ["ZZ", "XI"],
             "values": {"theta": 0.3}, "shots": 10, "seed": 7},
        )
        self.assertEqual(result["circuit_evaluations"], 1)
        self.assertEqual(result["total_shots"], 20)
        self.assertEqual(result["representation"], "state_vector")

    def test_noise_switches_to_density_matrix(self):
        result = self.svc.estimate_resources(
            variational_circuit(),
            exact_request(noise={"single_qubit_depolarizing": 0.1}),
        )
        self.assertEqual(result["representation"], "density_matrix")
        self.assertEqual(result["state_elements"], 16)
        self.assertEqual(result["state_bytes"], 256)

    def test_zero_probability_noise_stays_state_vector(self):
        result = self.svc.estimate_resources(
            variational_circuit(),
            exact_request(noise={"single_qubit_depolarizing": 0.0,
                                 "two_qubit_depolarizing": 0}),
        )
        self.assertEqual(result["representation"], "state_vector")
        self.assertEqual(result["state_elements"], 4)

    def test_gradient_counts_parameter_shift_evaluations(self):
        result = self.svc.estimate_resources(
            variational_circuit(),
            {"type": "gradient", "observables": ["ZZ"], "values": {"theta": 0.3}},
        )
        self.assertEqual(result["circuit_evaluations"], 3)  # 1 + 2r, r = 1
        self.assertEqual(result["gate_applications"], 9)

    def test_gradient_ignores_zero_coefficient_parameterizations(self):
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "theta", "coefficient": 0, "offset": 0.5}},
        ]}
        result = self.svc.estimate_resources(
            circuit, {"type": "gradient", "observables": ["Z"], "values": {"theta": 0.3}},
        )
        self.assertEqual(result["circuit_evaluations"], 1)
        self.assertEqual(result["gate_applications"], 1)

    def test_optimization_counts_worst_case_evaluations(self):
        result = self.svc.estimate_resources(variational_circuit(), optimization_request())
        # (max_iterations + 1) * (1 + 2r) = 6 * 3
        self.assertEqual(result["circuit_evaluations"], 18)
        self.assertEqual(result["gate_applications"], 54)
        self.assertEqual(result["total_shots"], 0)

    def test_result_is_deterministic(self):
        first = self.svc.estimate_resources(variational_circuit(), exact_request())
        second = self.svc.estimate_resources(variational_circuit(), exact_request())
        self.assertEqual(first, second)


class EstimateResourcesRuntimeLimitTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_state_vector_limit_is_20_qubits(self):
        request = {"type": "exact_expectation", "observables": ["I" * 21]}
        result = self.svc.estimate_resources({"qubit_count": 21}, request)
        self.assertFalse(result["runtime_supported"])
        self.assertFalse(result["admitted"])
        self.assertEqual(result["state_elements"], 1 << 21)

        result = self.svc.estimate_resources(
            {"qubit_count": 20}, {"type": "exact_expectation", "observables": ["I" * 20]},
        )
        self.assertTrue(result["runtime_supported"])
        self.assertTrue(result["admitted"])

    def test_density_matrix_limit_is_10_qubits(self):
        request = {"type": "exact_expectation", "observables": ["I" * 11],
                   "noise": {"two_qubit_depolarizing": 0.5}}
        result = self.svc.estimate_resources({"qubit_count": 11}, request)
        self.assertFalse(result["runtime_supported"])
        self.assertFalse(result["admitted"])
        self.assertEqual(result["state_elements"], 1 << 22)

    def test_oversized_state_space_does_not_raise(self):
        # 状态空间限制只写入结果，不抛 state_space_too_large。
        result = self.svc.estimate_resources(
            {"qubit_count": 30}, {"type": "exact_expectation", "observables": ["I" * 30]},
        )
        self.assertFalse(result["runtime_supported"])


class EstimateResourcesBudgetTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_omitted_budget_means_unlimited(self):
        result = self.svc.estimate_resources(variational_circuit(), exact_request())
        self.assertEqual(result["exceeded"], [])
        self.assertTrue(result["admitted"])

    def test_equal_to_budget_is_not_exceeded(self):
        result = self.svc.estimate_resources(
            variational_circuit(), exact_request(),
            budget={"max_state_bytes": 64, "max_circuit_evaluations": 1,
                    "max_gate_applications": 3, "max_total_shots": 0},
        )
        self.assertEqual(result["exceeded"], [])
        self.assertTrue(result["admitted"])

    def test_exceeded_keys_follow_canonical_order(self):
        budget = {"max_total_shots": 0, "max_gate_applications": 0,
                  "max_circuit_evaluations": 0, "max_state_bytes": 0}
        result = self.svc.estimate_resources(
            variational_circuit(), exact_request(), budget=budget,
        )
        self.assertEqual(result["exceeded"], [
            "max_state_bytes", "max_circuit_evaluations", "max_gate_applications",
        ])
        self.assertFalse(result["admitted"])
        self.assertTrue(result["runtime_supported"])

    def test_exceeded_does_not_raise(self):
        result = self.svc.estimate_resources(
            variational_circuit(), exact_request(), budget={"max_state_bytes": 1},
        )
        self.assertEqual(result["exceeded"], ["max_state_bytes"])

    def test_sampled_total_shots_budget(self):
        request = {"type": "sampled_expectation", "observables": ["ZZ", "XI"],
                   "values": {"theta": 0.3}, "shots": 10}
        result = self.svc.estimate_resources(
            variational_circuit(), request, budget={"max_total_shots": 19},
        )
        self.assertEqual(result["exceeded"], ["max_total_shots"])

    def test_invalid_budget_rejected(self):
        cases = [
            (5, "budget"),
            ({"max_shots": 1}, "budget.max_shots"),
            ({"max_state_bytes": -1}, "budget.max_state_bytes"),
            ({"max_state_bytes": True}, "budget.max_state_bytes"),
            ({"max_state_bytes": 1.5}, "budget.max_state_bytes"),
            ({"max_total_shots": None}, "budget.max_total_shots"),
        ]
        for budget, path in cases:
            with self.subTest(budget=budget):
                exc = res_err(self.svc.estimate_resources,
                              variational_circuit(), exact_request(), budget=budget)
                self.assertEqual(exc.code, "invalid_budget")
                self.assertEqual(exc.path, path)


class EstimateResourcesRequestValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_invalid_request_rejected(self):
        cases = [
            ("not-an-object", "request"),
            ({}, "request.type"),
            ({"type": "expectation"}, "request.type"),
            ({"type": 3}, "request.type"),
            # 无关字段非法：exact 不接受 shots/seed，gradient 不接受 terms。
            (exact_request(shots=10), "request.shots"),
            (exact_request(seed=1), "request.seed"),
            ({"type": "gradient", "observables": ["ZZ"], "terms": []}, "request.terms"),
            ({"type": "optimization", "terms": [], "values": {}, "config": {},
              "observables": ["ZZ"]}, "request.observables"),
            # 必填字段缺失。
            ({"type": "exact_expectation"}, "request.observables"),
            ({"type": "sampled_expectation", "observables": ["ZZ"]}, "request.shots"),
            ({"type": "optimization", "values": {}, "config": {}}, "request.terms"),
            ({"type": "optimization", "terms": [], "config": {}}, "request.values"),
            ({"type": "optimization", "terms": [], "values": {}}, "request.config"),
        ]
        for request, path in cases:
            with self.subTest(request=request):
                exc = res_err(self.svc.estimate_resources, variational_circuit(), request)
                self.assertEqual(exc.code, "invalid_resource_request")
                self.assertEqual(exc.path, path)


class EstimateResourcesEntryErrorTest(unittest.TestCase):
    """request/budget 之外的错误沿用对应入口的异常类型、code 与 path。"""

    def setUp(self):
        self.svc = Service()

    def test_circuit_errors(self):
        with self.assertRaises(CircuitValidationError) as ctx:
            self.svc.estimate_resources({"qubit_count": -1}, exact_request())
        self.assertEqual(ctx.exception.code, "invalid_value")
        self.assertEqual(ctx.exception.path, "qubit_count")

    def test_binding_errors(self):
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.estimate_resources(variational_circuit(), exact_request(values={}))
        self.assertEqual(ctx.exception.code, "missing_parameter")
        self.assertEqual(ctx.exception.path, "theta")

    def test_simulation_errors(self):
        with self.assertRaises(SimulationError) as ctx:
            self.svc.estimate_resources(variational_circuit(), exact_request(observables=["Z"]))
        self.assertEqual(ctx.exception.code, "invalid_observable")
        self.assertEqual(ctx.exception.path, "observables[0]")

        with self.assertRaises(SimulationError) as ctx:
            self.svc.estimate_resources(variational_circuit(), exact_request(noise={"p": 0.1}))
        self.assertEqual(ctx.exception.code, "invalid_noise_model")
        self.assertEqual(ctx.exception.path, "noise.p")

        request = {"type": "sampled_expectation", "observables": ["ZZ"],
                   "values": {"theta": 0.3}, "shots": 0}
        with self.assertRaises(SimulationError) as ctx:
            self.svc.estimate_resources(variational_circuit(), request)
        self.assertEqual(ctx.exception.code, "invalid_shots")
        self.assertEqual(ctx.exception.path, "shots")

    def test_optimization_errors(self):
        with self.assertRaises(OptimizationError) as ctx:
            self.svc.estimate_resources(
                variational_circuit(),
                optimization_request(terms=[{"observable": "ZZ"}]),
            )
        self.assertEqual(ctx.exception.code, "invalid_term")
        self.assertEqual(ctx.exception.path, "terms[0].coefficient")

        with self.assertRaises(OptimizationError) as ctx:
            self.svc.estimate_resources(
                variational_circuit(),
                optimization_request(config=gd_config(method="newton")),
            )
        self.assertEqual(ctx.exception.code, "unsupported_optimizer")
        self.assertEqual(ctx.exception.path, "config.method")

    def test_request_and_budget_checked_before_entry_validation(self):
        # 请求结构错误优先于电路错误；预算错误优先于入口语义错误。
        exc = res_err(self.svc.estimate_resources, {"qubit_count": -1}, {"type": "nope"})
        self.assertEqual(exc.code, "invalid_resource_request")

        exc = res_err(self.svc.estimate_resources,
                      {"qubit_count": -1}, exact_request(), budget={"max_state_bytes": -1})
        self.assertEqual(exc.code, "invalid_budget")


class EstimateResourcesImmutabilityTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_inputs_are_not_modified(self):
        circuit = variational_circuit()
        request = optimization_request()
        budget = {"max_state_bytes": 64}
        snapshot = copy.deepcopy((circuit, request, budget))
        self.svc.estimate_resources(circuit, request, budget=budget)
        self.assertEqual((circuit, request, budget), snapshot)


if __name__ == "__main__":
    unittest.main()
