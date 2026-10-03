import copy
import json
import unittest

from qubitfabric.service import (
    CircuitValidationError,
    OptimizationError,
    ParameterBindingError,
    ResourceEstimationError,
    Service,
    SimulationError,
)


def resource_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ResourceEstimationError as exc:
        return exc
    raise AssertionError("ResourceEstimationError not raised")


def expect_err(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} not raised")


def rx_rz_circuit():
    # 2 个参数化旋转（rx θ、rz φ）+ 1 个常量 rx + cx，共 4 个操作；r = 2。
    return {"qubit_count": 2, "parameters": ["theta", "phi"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        {"gate": "rz", "target": 1, "angle": {"parameter": "phi", "coefficient": 2.0}},
        {"gate": "rx", "target": 0, "angle": 0.5},
        {"gate": "cx", "control": 0, "target": 1},
    ]}


def gd_config(**overrides):
    config = {"method": "gradient_descent", "learning_rate": 0.5,
              "max_iterations": 10, "tolerance": 1e-9}
    config.update(overrides)
    return config


VALUES = {"theta": 0.1, "phi": 0.2}


class ExactExpectationEstimateTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_rz_circuit()

    def test_state_vector_accounting(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "exact_expectation", "observables": ["ZZ", "IX"], "values": VALUES},
        )
        self.assertEqual(result["qubit_count"], 2)
        self.assertEqual(result["type"], "exact_expectation")
        self.assertEqual(result["representation"], "state_vector")
        self.assertEqual(result["state_elements"], 4)
        self.assertEqual(result["state_bytes"], 64)
        self.assertEqual(result["circuit_evaluations"], 1)
        self.assertEqual(result["gate_applications"], 4)
        self.assertEqual(result["total_shots"], 0)
        self.assertTrue(result["runtime_supported"])
        self.assertEqual(result["exceeded"], [])
        self.assertTrue(result["admitted"])

    def test_noise_switches_to_density_matrix(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "exact_expectation", "observables": ["ZZ"], "values": VALUES,
             "noise": {"single_qubit_depolarizing": 0.1}},
        )
        self.assertEqual(result["representation"], "density_matrix")
        self.assertEqual(result["state_elements"], 16)
        self.assertEqual(result["state_bytes"], 256)
        self.assertEqual(result["total_shots"], 0)
        self.assertTrue(result["runtime_supported"])

    def test_zero_probability_noise_stays_state_vector(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "exact_expectation", "observables": ["ZZ"],
             "values": VALUES, "noise": {}},
        )
        self.assertEqual(result["representation"], "state_vector")


class SampledExpectationEstimateTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_rz_circuit()

    def test_total_shots_is_shots_times_observable_count(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "sampled_expectation", "observables": ["ZZ", "IX", "YY"],
             "values": VALUES, "shots": 100, "seed": 7},
        )
        self.assertEqual(result["type"], "sampled_expectation")
        self.assertEqual(result["circuit_evaluations"], 1)
        self.assertEqual(result["gate_applications"], 4)
        self.assertEqual(result["total_shots"], 300)
        self.assertTrue(result["admitted"])

    def test_seed_optional(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "sampled_expectation", "observables": ["ZZ"],
             "values": VALUES, "shots": 10},
        )
        self.assertEqual(result["total_shots"], 10)

    def test_invalid_shots_uses_entrypoint_error(self):
        exc = expect_err(
            SimulationError,
            self.svc.estimate_resources,
            self.circuit,
            {"type": "sampled_expectation", "observables": ["ZZ"],
             "values": VALUES, "shots": 0},
        )
        self.assertEqual(exc.code, "invalid_shots")
        self.assertEqual(exc.path, "shots")


class GradientEstimateTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_rz_circuit()

    def test_gradient_evaluations_are_1_plus_2r(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "gradient", "observables": ["ZZ", "IX"], "values": VALUES},
        )
        self.assertEqual(result["type"], "gradient")
        self.assertEqual(result["circuit_evaluations"], 5)  # 1 + 2*2
        self.assertEqual(result["gate_applications"], 20)
        self.assertEqual(result["total_shots"], 0)

    def test_zero_coefficient_parameter_angle_is_not_parameterized(self):
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0,
             "angle": {"parameter": "theta", "coefficient": 0.0, "offset": 0.3}},
        ]}
        result = self.svc.estimate_resources(
            circuit, {"type": "gradient", "observables": ["Z"], "values": {"theta": 0.0}},
        )
        self.assertEqual(result["circuit_evaluations"], 1)
        self.assertEqual(result["gate_applications"], 1)


class OptimizationEstimateTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_rz_circuit()

    def request(self, **overrides):
        request = {
            "type": "optimization",
            "terms": [{"observable": "ZZ", "coefficient": 1.0},
                      {"observable": "IX", "coefficient": -0.5}],
            "values": {"theta": 0.1, "phi": 0.2},
            "config": gd_config(),
        }
        request.update(overrides)
        return request

    def test_optimization_worst_case_evaluations(self):
        result = self.svc.estimate_resources(self.circuit, self.request())
        self.assertEqual(result["type"], "optimization")
        self.assertEqual(result["representation"], "state_vector")
        # (10 + 1) * (1 + 2*2) = 55
        self.assertEqual(result["circuit_evaluations"], 55)
        self.assertEqual(result["gate_applications"], 55 * 4)
        self.assertEqual(result["total_shots"], 0)
        self.assertTrue(result["admitted"])

    def test_config_max_iterations_drives_count(self):
        result = self.svc.estimate_resources(
            self.circuit, self.request(config=gd_config(max_iterations=3)),
        )
        self.assertEqual(result["circuit_evaluations"], 4 * 5)

    def test_invalid_config_uses_optimization_error(self):
        bad = self.request(config=gd_config(method="bogus"))
        exc = expect_err(OptimizationError, self.svc.estimate_resources, self.circuit, bad)
        self.assertEqual(exc.code, "unsupported_optimizer")
        self.assertEqual(exc.path, "config.method")

    def test_missing_values_binding_error(self):
        exc = expect_err(
            ParameterBindingError,
            self.svc.estimate_resources,
            self.circuit,
            self.request(values={"theta": 0.1}),
        )
        self.assertEqual(exc.code, "missing_parameter")
        self.assertEqual(exc.path, "phi")


class RuntimeSupportTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_state_vector_limit_is_20_qubits(self):
        circuit = {"qubit_count": 20, "operations": []}
        ok = self.svc.estimate_resources(
            circuit, {"type": "exact_expectation", "observables": ["I" * 20]},
        )
        self.assertTrue(ok["runtime_supported"])
        self.assertEqual(ok["state_elements"], 2 ** 20)
        self.assertTrue(ok["admitted"])

        circuit = {"qubit_count": 21, "operations": []}
        too_big = self.svc.estimate_resources(
            circuit, {"type": "exact_expectation", "observables": ["I" * 21]},
        )
        self.assertFalse(too_big["runtime_supported"])
        self.assertEqual(too_big["state_elements"], 2 ** 21)
        self.assertFalse(too_big["admitted"])
        self.assertEqual(too_big["exceeded"], [])

    def test_density_matrix_limit_is_10_qubits(self):
        noise = {"two_qubit_depolarizing": 0.01}
        circuit = {"qubit_count": 10, "operations": []}
        ok = self.svc.estimate_resources(
            circuit,
            {"type": "exact_expectation", "observables": ["I" * 10], "noise": noise},
        )
        self.assertTrue(ok["runtime_supported"])
        self.assertEqual(ok["state_elements"], 4 ** 10)

        circuit = {"qubit_count": 11, "operations": []}
        too_big = self.svc.estimate_resources(
            circuit,
            {"type": "exact_expectation", "observables": ["I" * 11], "noise": noise},
        )
        self.assertEqual(too_big["representation"], "density_matrix")
        self.assertFalse(too_big["runtime_supported"])
        self.assertFalse(too_big["admitted"])


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_rz_circuit()

    def test_exceeded_in_declared_order(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "gradient", "observables": ["ZZ", "IX"], "values": VALUES},
            {
                "max_state_bytes": 1,            # 64 > 1
                "max_circuit_evaluations": 1,   # 5 > 1
                "max_gate_applications": 1,     # 20 > 1
                "max_total_shots": 1,           # 0 不超
            },
        )
        self.assertEqual(
            result["exceeded"],
            ["max_state_bytes", "max_circuit_evaluations", "max_gate_applications"],
        )
        self.assertFalse(result["admitted"])

    def test_boundary_equal_is_not_exceeded(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "exact_expectation", "observables": ["ZZ"], "values": VALUES},
            {"max_state_bytes": 64, "max_circuit_evaluations": 1,
             "max_gate_applications": 4, "max_total_shots": 0},
        )
        self.assertEqual(result["exceeded"], [])
        self.assertTrue(result["admitted"])

    def test_sampling_budget(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "sampled_expectation", "observables": ["ZZ", "IX"],
             "values": VALUES, "shots": 50},
            {"max_total_shots": 99},
        )
        self.assertEqual(result["exceeded"], ["max_total_shots"])
        self.assertFalse(result["admitted"])

    def test_zero_budget_values_allowed(self):
        result = self.svc.estimate_resources(
            {"qubit_count": 0, "operations": []},
            {"type": "exact_expectation", "observables": [""]},
            {"max_state_bytes": 0, "max_gate_applications": 0},
        )
        # 空电路：1 元素 * 16 = 16 字节 > 0；gate_applications = 0。
        self.assertEqual(result["state_bytes"], 16)
        self.assertEqual(result["exceeded"], ["max_state_bytes"])

    def test_partial_budget_only_limits_provided_keys(self):
        result = self.svc.estimate_resources(
            self.circuit,
            {"type": "exact_expectation", "observables": ["ZZ"], "values": VALUES},
            {"max_gate_applications": 100},
        )
        self.assertEqual(result["exceeded"], [])
        self.assertTrue(result["admitted"])

    def test_unsupported_runtime_still_reports_budget_exceeded(self):
        circuit = {"qubit_count": 21, "operations": []}
        result = self.svc.estimate_resources(
            circuit,
            {"type": "exact_expectation", "observables": ["I" * 21]},
            {"max_state_bytes": 10},
        )
        self.assertFalse(result["runtime_supported"])
        self.assertEqual(result["exceeded"], ["max_state_bytes"])
        self.assertFalse(result["admitted"])


class InvalidBudgetTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_rz_circuit()
        self.request = {"type": "exact_expectation", "observables": ["ZZ"],
                        "values": VALUES}

    def test_budget_not_object(self):
        exc = resource_err(self.svc.estimate_resources, self.circuit, self.request, [])
        self.assertEqual(exc.code, "invalid_budget")
        self.assertEqual(exc.path, "$")

    def test_unknown_budget_field(self):
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, self.request, {"max_qubits": 5},
        )
        self.assertEqual(exc.code, "invalid_budget")
        self.assertEqual(exc.path, "max_qubits")

    def test_boolean_value_rejected(self):
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, self.request,
            {"max_state_bytes": True},
        )
        self.assertEqual(exc.code, "invalid_budget")
        self.assertEqual(exc.path, "max_state_bytes")

    def test_negative_value_rejected(self):
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, self.request,
            {"max_circuit_evaluations": -1},
        )
        self.assertEqual(exc.code, "invalid_budget")
        self.assertEqual(exc.path, "max_circuit_evaluations")

    def test_float_value_rejected(self):
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, self.request,
            {"max_total_shots": 1.5},
        )
        self.assertEqual(exc.code, "invalid_budget")
        self.assertEqual(exc.path, "max_total_shots")

    def test_budget_validated_after_request(self):
        # 非法 request 与非法 budget 同时存在时先报 request。
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, {"type": "bogus"}, {"nope": 1},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "type")


class InvalidRequestTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_rz_circuit()

    def test_request_not_object(self):
        exc = resource_err(self.svc.estimate_resources, self.circuit, "gradient")
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "$")

    def test_missing_type(self):
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, {"observables": ["ZZ"]},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "type")

    def test_unknown_type(self):
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, {"type": "vqe"},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "type")

    def test_irrelevant_field_rejected(self):
        exc = resource_err(
            self.svc.estimate_resources,
            self.circuit,
            {"type": "exact_expectation", "observables": ["ZZ"], "shots": 10},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "shots")

    def test_gradient_rejects_shots(self):
        exc = resource_err(
            self.svc.estimate_resources,
            self.circuit,
            {"type": "gradient", "observables": ["ZZ"], "shots": 10},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "shots")

    def test_optimization_rejects_observables(self):
        exc = resource_err(
            self.svc.estimate_resources,
            self.circuit,
            {"type": "optimization", "observables": ["ZZ"]},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "observables")

    def test_missing_observables(self):
        exc = resource_err(
            self.svc.estimate_resources, self.circuit, {"type": "gradient"},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "observables")

    def test_missing_shots(self):
        exc = resource_err(
            self.svc.estimate_resources,
            self.circuit,
            {"type": "sampled_expectation", "observables": ["ZZ"]},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "shots")

    def test_missing_config(self):
        exc = resource_err(
            self.svc.estimate_resources,
            self.circuit,
            {"type": "optimization",
             "terms": [{"observable": "ZZ", "coefficient": 1.0}]},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "config")


class EntrypointValidationOrderTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_circuit_error_precedes_request_content(self):
        bad_circuit = {"qubit_count": 1, "operations": [{"gate": "nope", "target": 0}]}
        exc = expect_err(
            CircuitValidationError,
            self.svc.estimate_resources,
            bad_circuit,
            {"type": "exact_expectation", "observables": ["Z"]},
        )
        self.assertEqual(exc.code, "unknown_gate")
        self.assertEqual(exc.path, "operations[0].gate")

    def test_observable_content_uses_simulation_error(self):
        circuit = {"qubit_count": 2, "operations": []}
        exc = expect_err(
            SimulationError,
            self.svc.estimate_resources,
            circuit,
            {"type": "exact_expectation", "observables": ["ZZZ"]},
        )
        self.assertEqual(exc.code, "invalid_observable")
        self.assertEqual(exc.path, "observables[0]")

    def test_noise_error_uses_simulation_error(self):
        circuit = {"qubit_count": 1, "operations": []}
        exc = expect_err(
            SimulationError,
            self.svc.estimate_resources,
            circuit,
            {"type": "gradient", "observables": ["Z"],
             "noise": {"single_qubit_depolarizing": 2}},
        )
        self.assertEqual(exc.code, "invalid_noise_model")
        self.assertEqual(exc.path, "noise.single_qubit_depolarizing")

    def test_terms_structure_uses_optimization_error(self):
        circuit = rx_rz_circuit()
        request = {"type": "optimization", "terms": [], "config": gd_config(),
                   "values": VALUES}
        exc = expect_err(OptimizationError, self.svc.estimate_resources, circuit, request)
        self.assertEqual(exc.code, "invalid_terms")
        self.assertEqual(exc.path, "terms")

    def test_state_space_limit_does_not_raise_in_resource_entry(self):
        # 计算入口会抛 state_space_too_large；准入只写 runtime_supported。
        circuit = {"qubit_count": 25, "operations": []}
        result = self.svc.estimate_resources(
            circuit, {"type": "gradient", "observables": ["I" * 25]},
        )
        self.assertFalse(result["runtime_supported"])
        self.assertFalse(result["admitted"])

    def test_null_observables_delegates_to_simulation_error(self):
        circuit = {"qubit_count": 1, "operations": []}
        exc = expect_err(
            SimulationError,
            self.svc.estimate_resources,
            circuit,
            {"type": "exact_expectation", "observables": None},
        )
        self.assertEqual(exc.code, "invalid_observables")
        self.assertEqual(exc.path, "observables")

    def test_null_terms_delegates_to_optimization_error(self):
        circuit = rx_rz_circuit()
        exc = expect_err(
            OptimizationError,
            self.svc.estimate_resources,
            circuit,
            {"type": "optimization", "terms": None, "config": gd_config(),
             "values": VALUES},
        )
        self.assertEqual(exc.code, "invalid_terms")
        self.assertEqual(exc.path, "terms")

    def test_null_shots_in_sampled_request_is_invalid_resource_request(self):
        circuit = {"qubit_count": 1, "operations": []}
        exc = resource_err(
            self.svc.estimate_resources,
            circuit,
            {"type": "sampled_expectation", "observables": ["Z"], "shots": None},
        )
        self.assertEqual(exc.code, "invalid_resource_request")
        self.assertEqual(exc.path, "shots")


class ImmutabilityDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_inputs_are_not_modified(self):
        circuit = rx_rz_circuit()
        request = {"type": "sampled_expectation", "observables": ["ZZ", "IX"],
                   "values": VALUES, "shots": 50, "seed": 1,
                   "noise": {"single_qubit_depolarizing": 0.2}}
        budget = {"max_state_bytes": 10 ** 9, "max_total_shots": 1000}
        circuit_snapshot = copy.deepcopy(circuit)
        request_snapshot = copy.deepcopy(request)
        budget_snapshot = copy.deepcopy(budget)

        self.svc.estimate_resources(circuit, request, budget)
        self.assertEqual(circuit, circuit_snapshot)
        self.assertEqual(request, request_snapshot)
        self.assertEqual(budget, budget_snapshot)

    def test_result_is_json_serializable_and_deterministic(self):
        circuit = rx_rz_circuit()
        request = {"type": "optimization",
                   "terms": [{"observable": "ZZ", "coefficient": 1.0}],
                   "values": {"theta": 0.0, "phi": 0.0},
                   "config": gd_config(),
                   "noise": {"single_qubit_depolarizing": 0.1}}
        budget = {"max_circuit_evaluations": 10}
        first = self.svc.estimate_resources(circuit, request, budget)
        second = self.svc.estimate_resources(copy.deepcopy(circuit),
                                             copy.deepcopy(request),
                                             copy.deepcopy(budget))
        self.assertEqual(first, second)
        # 全部为 JSON 原生类型。
        json.loads(json.dumps(first))
        self.assertFalse(any(key not in first for key in (
            "qubit_count", "type", "representation", "state_elements", "state_bytes",
            "circuit_evaluations", "gate_applications", "total_shots",
            "runtime_supported", "admitted", "exceeded",
        )))


if __name__ == "__main__":
    unittest.main()
