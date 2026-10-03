import copy
import json
import math
import os
import subprocess
import sys
import unittest

from qubitfabric import RuntimeStateError as PackageRuntimeStateError
from qubitfabric.service import (
    CircuitValidationError,
    OptimizationError,
    ParameterBindingError,
    RuntimeStateError,
    Service,
    SimulationError,
)


def rx_circuit():
    return {"qubit_count": 1, "parameters": ["theta"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
    ]}


def two_param_circuit():
    return {"qubit_count": 2, "parameters": ["alpha", "beta"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "alpha"}},
        {"gate": "rz", "target": 1, "angle": {"parameter": "beta", "coefficient": 2.0}},
        {"gate": "cx", "control": 0, "target": 1},
    ]}


def gd_config(**overrides):
    config = {"method": "gradient_descent", "learning_rate": 0.5,
              "max_iterations": 25, "tolerance": 1e-9}
    config.update(overrides)
    return config


def state_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except RuntimeStateError as exc:
        return exc
    raise AssertionError("RuntimeStateError not raised")


class ResumableSegmentationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.terms = [{"observable": "Z", "coefficient": 1.0}]
        self.values = {"theta": 0.3}

    def run_segments(self, budgets, circuit=None, terms=None, values=None,
                     config=None, noise=None, roundtrip=True):
        circuit = self.circuit if circuit is None else circuit
        terms = self.terms if terms is None else terms
        values = self.values if values is None else values
        config = gd_config() if config is None else config
        checkpoint = None
        response = None
        for budget in budgets:
            response = self.svc.optimize_resumable(
                circuit, terms, values, config,
                noise=noise, step_budget=budget, checkpoint=checkpoint,
            )
            json.dumps(response)
            if response["status"] == "completed":
                return response
            checkpoint = response["checkpoint"]
            if roundtrip:
                checkpoint = json.loads(json.dumps(checkpoint))
        self.fail("did not complete within the given budgets")

    def test_single_large_budget_matches_optimize(self):
        config = gd_config()
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        response = self.run_segments([100], config=config)
        self.assertEqual(response["status"], "completed")
        self.assertIsNone(response["checkpoint"])
        self.assertEqual(response["result"], reference)

    def test_unit_budget_segments_match_optimize(self):
        config = gd_config()
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        response = self.run_segments([1] * config["max_iterations"], config=config)
        self.assertEqual(response["result"], reference)

    def test_mixed_budgets_match_optimize(self):
        config = gd_config()
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        for budgets in ([2, 3, 1, 4, 50], [7, 7, 7, 7], [25]):
            response = self.run_segments(list(budgets), config=config)
            self.assertEqual(response["result"], reference)

    def test_adam_segments_match_optimize(self):
        config = gd_config(method="adam", learning_rate=0.2, max_iterations=30)
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        response = self.run_segments([1, 2, 3, 4, 5, 100], config=config)
        self.assertEqual(response["result"], reference)

    def test_multi_parameter_segments_match_optimize(self):
        circuit = two_param_circuit()
        terms = [
            {"observable": "ZI", "coefficient": 1.5},
            {"observable": "IZ", "coefficient": -0.5},
        ]
        values = {"alpha": 0.4, "beta": -0.2}
        config = gd_config(method="adam", learning_rate=0.1)
        reference = self.svc.optimize(circuit, terms, values, config)
        response = self.run_segments([3, 1, 2, 100], circuit=circuit, terms=terms,
                                     values=values, config=config)
        self.assertEqual(response["result"], reference)
        self.assertEqual(response["result"]["parameters"], ["alpha", "beta"])

    def test_noise_segments_match_optimize(self):
        noise = {"single_qubit_depolarizing": 0.05, "two_qubit_depolarizing": 0.01}
        config = gd_config()
        reference = self.svc.optimize(self.circuit, self.terms, self.values,
                                      config, noise=noise)
        response = self.run_segments([2, 2, 100], config=config, noise=noise)
        self.assertEqual(response["result"], reference)

    def test_paused_response_shape_and_progress(self):
        config = gd_config()
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        response = self.svc.optimize_resumable(
            self.circuit, self.terms, self.values, config, step_budget=1,
        )
        self.assertEqual(response["status"], "paused")
        self.assertIsNone(response["result"])
        self.assertIsInstance(response["checkpoint"], dict)
        progress = response["progress"]
        self.assertEqual(progress["iterations"], 1)
        self.assertEqual(progress["parameters"], ["theta"])
        self.assertEqual(progress["values"], reference["history"][1]["values"])
        self.assertEqual(progress["objective"], reference["history"][1]["objective"])
        self.assertEqual(progress["gradient_norm"], reference["history"][1]["gradient_norm"])
        self.assertEqual(progress["history"], reference["history"][:2])
        self.assertEqual([e["iteration"] for e in progress["history"]], [0, 1])

    def test_first_call_produces_iteration_zero_evaluation(self):
        config = gd_config()
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        response = self.svc.optimize_resumable(
            self.circuit, self.terms, self.values, config, step_budget=1,
        )
        self.assertEqual(response["progress"]["history"][0], reference["history"][0])

    def test_initial_point_already_converged_completes_immediately(self):
        config = gd_config()
        values = {"theta": math.pi}
        reference = self.svc.optimize(self.circuit, self.terms, values, config)
        self.assertEqual(reference["iterations"], 0)
        response = self.svc.optimize_resumable(
            self.circuit, self.terms, values, config, step_budget=1,
        )
        self.assertEqual(response["status"], "completed")
        self.assertIsNone(response["checkpoint"])
        self.assertEqual(response["result"], reference)

    def test_max_iterations_boundary_completes(self):
        config = gd_config(max_iterations=4, tolerance=0.0)
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        self.assertFalse(reference["converged"])
        response = self.run_segments([2, 2], config=config)
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["result"], reference)

    def test_checkpoint_json_roundtrip_across_processes(self):
        config = gd_config()
        first = self.svc.optimize_resumable(
            self.circuit, self.terms, self.values, config, step_budget=2,
        )
        self.assertEqual(first["status"], "paused")
        checkpoint_json = json.dumps(first["checkpoint"])

        src = os.path.join(os.path.dirname(__file__), "..", "src")
        script = (
            "import json, sys\n"
            "from qubitfabric.service import Service\n"
            "circuit, terms, values, config, checkpoint = json.load(sys.stdin)\n"
            "response = Service().optimize_resumable(\n"
            "    circuit, terms, values, config, step_budget=100, checkpoint=checkpoint)\n"
            "json.dump(response, sys.stdout)\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.abspath(src)
        payload = json.dumps([self.circuit, self.terms, self.values, config,
                              json.loads(checkpoint_json)])
        proc = subprocess.run(
            [sys.executable, "-c", script], input=payload,
            capture_output=True, text=True, env=env, check=True,
        )
        resumed = json.loads(proc.stdout)
        reference = self.svc.optimize(self.circuit, self.terms, self.values, config)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["result"], reference)

    def test_inputs_and_checkpoint_not_mutated(self):
        circuit = two_param_circuit()
        terms = [{"observable": "ZI", "coefficient": 1.5}]
        values = {"alpha": 0.4, "beta": -0.2}
        config = gd_config()
        noise = {"single_qubit_depolarizing": 0.1}
        snapshot = copy.deepcopy((circuit, terms, values, config, noise))

        first = self.svc.optimize_resumable(
            circuit, terms, values, config, noise=noise, step_budget=1,
        )
        checkpoint = first["checkpoint"]
        checkpoint_snapshot = copy.deepcopy(checkpoint)
        self.svc.optimize_resumable(
            circuit, terms, values, config, noise=noise,
            step_budget=100, checkpoint=checkpoint,
        )
        self.assertEqual((circuit, terms, values, config, noise), snapshot)
        self.assertEqual(checkpoint, checkpoint_snapshot)


class StepBudgetValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.args = (rx_circuit(), [{"observable": "Z", "coefficient": 1.0}],
                     {"theta": 0.3}, gd_config())

    def test_bool_is_rejected(self):
        for bad in (True, False):
            exc = state_err(self.svc.optimize_resumable, *self.args, step_budget=bad)
            self.assertEqual(exc.code, "invalid_step_budget")
            self.assertEqual(exc.path, "step_budget")

    def test_non_integer_is_rejected(self):
        for bad in (1.5, "3", [1], {"n": 1}, None, float("nan")):
            exc = state_err(self.svc.optimize_resumable, *self.args, step_budget=bad)
            self.assertEqual(exc.code, "invalid_step_budget")
            self.assertEqual(exc.path, "step_budget")

    def test_less_than_one_is_rejected(self):
        for bad in (0, -1, -100):
            exc = state_err(self.svc.optimize_resumable, *self.args, step_budget=bad)
            self.assertEqual(exc.code, "invalid_step_budget")
            self.assertEqual(exc.path, "step_budget")

    def test_float_whole_number_is_rejected(self):
        exc = state_err(self.svc.optimize_resumable, *self.args, step_budget=3.0)
        self.assertEqual(exc.code, "invalid_step_budget")


class CheckpointValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.terms = [{"observable": "Z", "coefficient": 1.0}]
        self.values = {"theta": 0.3}
        self.config = gd_config()
        paused = self.svc.optimize_resumable(
            self.circuit, self.terms, self.values, self.config, step_budget=2,
        )
        self.assertEqual(paused["status"], "paused")
        self.checkpoint = paused["checkpoint"]

    def resume(self, checkpoint, **overrides):
        config = overrides.pop("config", self.config)
        values = overrides.pop("values", self.values)
        terms = overrides.pop("terms", self.terms)
        circuit = overrides.pop("circuit", self.circuit)
        noise = overrides.pop("noise", None)
        return self.svc.optimize_resumable(
            circuit, terms, values, config,
            noise=noise, step_budget=1, checkpoint=checkpoint,
        )

    def assert_invalid(self, checkpoint, **overrides):
        exc = state_err(self.resume, checkpoint, **overrides)
        self.assertEqual(exc.code, "invalid_checkpoint")
        self.assertEqual(exc.path, "checkpoint")
        return exc

    def assert_mismatch(self, checkpoint, **overrides):
        exc = state_err(self.resume, checkpoint, **overrides)
        self.assertEqual(exc.code, "checkpoint_mismatch")
        self.assertEqual(exc.path, "checkpoint")
        return exc

    def tweaked(self, **changes):
        checkpoint = copy.deepcopy(self.checkpoint)
        checkpoint.update(changes)
        return checkpoint

    def test_not_an_object(self):
        for bad in ([], "checkpoint", 3, 1.5, True, None):
            # None 表示无检查点，单独语义；其余都不是对象。
            if bad is None:
                continue
            self.assert_invalid(bad)

    def test_unknown_field(self):
        self.assert_invalid(self.tweaked(extra=1))

    def test_missing_field(self):
        for field in self.checkpoint:
            checkpoint = copy.deepcopy(self.checkpoint)
            del checkpoint[field]
            self.assert_invalid(checkpoint)

    def test_unsupported_version(self):
        for bad in (2, 0, "1", 1.0, True, None):
            self.assert_invalid(self.tweaked(version=bad))

    def test_field_type_errors(self):
        self.assert_invalid(self.tweaked(fingerprint=123))
        self.assert_invalid(self.tweaked(iterations=1.5))
        self.assert_invalid(self.tweaked(iterations=True))
        self.assert_invalid(self.tweaked(iterations=-1))
        self.assert_invalid(self.tweaked(values=[0.1]))
        self.assert_invalid(self.tweaked(values={"theta": "x"}))
        self.assert_invalid(self.tweaked(objective="x"))
        self.assert_invalid(self.tweaked(gradient_norm=-0.5))
        self.assert_invalid(self.tweaked(history={}))
        self.assert_invalid(self.tweaked(history=[]))

    def test_non_finite_numbers(self):
        self.assert_invalid(self.tweaked(objective=float("nan")))
        self.assert_invalid(self.tweaked(gradient_norm=float("inf")))
        self.assert_invalid(self.tweaked(values={"theta": float("nan")}))
        self.assert_invalid(self.tweaked(gradients={"theta": float("-inf")}))
        checkpoint = self.tweaked()
        checkpoint["history"][0]["objective"] = float("nan")
        self.assert_invalid(checkpoint)

    def test_state_map_key_mismatch(self):
        self.assert_invalid(self.tweaked(gradients={"other": 0.1}))
        self.assert_invalid(self.tweaked(first_moment={}))
        checkpoint = self.tweaked()
        checkpoint["history"][1]["values"] = {"other": 0.1}
        self.assert_invalid(checkpoint)

    def test_history_contradictions(self):
        self.assert_invalid(self.tweaked(iterations=5))
        checkpoint = self.tweaked()
        checkpoint["history"][1]["iteration"] = 7
        self.assert_invalid(checkpoint)
        checkpoint = self.tweaked(values={"theta": 9.9})
        self.assert_invalid(checkpoint)
        checkpoint = self.tweaked(objective=42.0)
        self.assert_invalid(checkpoint)
        checkpoint = self.tweaked()
        checkpoint["history"][1]["gradient_norm"] = 0.0
        self.assert_invalid(checkpoint)

    def test_converged_state_is_contradictory(self):
        checkpoint = self.tweaked(gradient_norm=0.0)
        checkpoint["history"][-1]["gradient_norm"] = 0.0
        self.assert_invalid(checkpoint)

    def test_exhausted_state_is_contradictory(self):
        # 人为构造 iterations 达到 max_iterations 的“暂停”状态。
        checkpoint = self.tweaked(iterations=self.config["max_iterations"])
        last = checkpoint["history"][-1]
        checkpoint["history"] = [
            {"iteration": i, "values": dict(last["values"]),
             "objective": last["objective"], "gradient_norm": last["gradient_norm"]}
            for i in range(self.config["max_iterations"] + 1)
        ]
        self.assert_invalid(checkpoint)

    def test_mismatch_on_different_inputs(self):
        checkpoint = copy.deepcopy(self.checkpoint)
        self.assert_mismatch(checkpoint, values={"theta": 0.4})
        self.assert_mismatch(checkpoint, config=gd_config(learning_rate=0.4))
        self.assert_mismatch(checkpoint, config=gd_config(method="adam"))
        self.assert_mismatch(checkpoint, terms=[{"observable": "Z", "coefficient": 2.0}])
        self.assert_mismatch(checkpoint, terms=[{"observable": "X", "coefficient": 1.0}])
        self.assert_mismatch(checkpoint, noise={"single_qubit_depolarizing": 0.1})
        other_circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        self.assert_mismatch(checkpoint, circuit=other_circuit)

    def test_foreign_checkpoint_is_mismatch_not_invalid(self):
        # 另一问题产生的合法检查点：结构完全合法，只是指纹不符。
        other_values = {"theta": 0.9}
        paused = self.svc.optimize_resumable(
            self.circuit, self.terms, other_values, self.config, step_budget=2,
        )
        self.assert_mismatch(paused["checkpoint"])

    def test_forged_fingerprint_still_validates_state(self):
        # 指纹一致但参数集与电路矛盾：状态矛盾而非 mismatch。
        checkpoint = self.tweaked(values={"theta": 0.1, "extra": 0.2})
        checkpoint["gradients"] = {"theta": 0.1, "extra": 0.2}
        checkpoint["first_moment"] = {"theta": 0.0, "extra": 0.0}
        checkpoint["second_moment"] = {"theta": 0.0, "extra": 0.0}
        checkpoint["history"][-1]["values"] = {"theta": 0.1, "extra": 0.2}
        self.assert_invalid(checkpoint)

    def test_valid_checkpoint_resumes(self):
        response = self.resume(json.loads(json.dumps(self.checkpoint)),
                               config=gd_config())
        # config 相同，预算 1 继续推进或结束，二者都合法。
        self.assertIn(response["status"], ("paused", "completed"))


class ValidationOrderTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.terms = [{"observable": "Z", "coefficient": 1.0}]
        self.values = {"theta": 0.3}
        self.config = gd_config()

    def test_base_errors_precede_budget_and_checkpoint(self):
        bad_checkpoint = {"version": 99}
        with self.assertRaises(CircuitValidationError):
            self.svc.optimize_resumable(
                {"qubit_count": -1}, self.terms, self.values, self.config,
                step_budget=0, checkpoint=bad_checkpoint,
            )
        with self.assertRaises(ParameterBindingError):
            self.svc.optimize_resumable(
                self.circuit, self.terms, {}, self.config,
                step_budget=0, checkpoint=bad_checkpoint,
            )
        with self.assertRaises(OptimizationError):
            self.svc.optimize_resumable(
                self.circuit, [], self.values, self.config,
                step_budget=0, checkpoint=bad_checkpoint,
            )
        with self.assertRaises(SimulationError):
            self.svc.optimize_resumable(
                self.circuit, [{"observable": "Q", "coefficient": 1.0}],
                self.values, self.config, step_budget=0, checkpoint=bad_checkpoint,
            )
        with self.assertRaises(SimulationError):
            self.svc.optimize_resumable(
                self.circuit, self.terms, self.values, self.config,
                noise={"single_qubit_depolarizing": 2.0},
                step_budget=0, checkpoint=bad_checkpoint,
            )
        with self.assertRaises(OptimizationError):
            self.svc.optimize_resumable(
                self.circuit, self.terms, self.values, {"method": "nope"},
                step_budget=0, checkpoint=bad_checkpoint,
            )

    def test_budget_precedes_checkpoint(self):
        exc = state_err(
            self.svc.optimize_resumable,
            self.circuit, self.terms, self.values, self.config,
            step_budget=0, checkpoint={"version": 99},
        )
        self.assertEqual(exc.code, "invalid_step_budget")

    def test_runtime_state_error_is_value_error_and_exported(self):
        self.assertIs(PackageRuntimeStateError, RuntimeStateError)
        self.assertTrue(issubclass(RuntimeStateError, ValueError))
        exc = state_err(
            self.svc.optimize_resumable,
            self.circuit, self.terms, self.values, self.config, step_budget=0,
        )
        self.assertEqual(exc.code, "invalid_step_budget")
        self.assertEqual(exc.path, "step_budget")
        self.assertEqual(str(RuntimeStateError("x", "y")), "x at y")


if __name__ == "__main__":
    unittest.main()
