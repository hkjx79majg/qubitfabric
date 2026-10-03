import copy
import json
import math
import unittest

from qubitfabric import RuntimeStateError as PackageRuntimeStateError
from qubitfabric.service import (
    OptimizationError,
    RuntimeStateError,
    Service,
)


def rx_circuit():
    return {"qubit_count": 1, "parameters": ["theta"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
    ]}


def two_param_circuit():
    return {"qubit_count": 2, "parameters": ["a", "b"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "a"}},
        {"gate": "rx", "target": 1, "angle": {"parameter": "b"}},
    ]}


def gd_config(**overrides):
    config = {"method": "gradient_descent", "learning_rate": 0.5,
              "max_iterations": 50, "tolerance": 1e-12}
    config.update(overrides)
    return config


def adam_config(**overrides):
    config = {"method": "adam", "learning_rate": 0.2, "max_iterations": 60,
              "tolerance": 1e-10, "beta1": 0.8, "beta2": 0.9, "epsilon": 1e-6}
    config.update(overrides)
    return config


def state_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except RuntimeStateError as exc:
        return exc
    raise AssertionError("RuntimeStateError not raised")


class ResumableSegmentationTest(unittest.TestCase):
    """任意分段与单次 optimize 的数值、参数顺序和 history 完全一致。"""

    def setUp(self):
        self.svc = Service()

    def run_segmented(self, circuit, terms, values, config, budgets, noise=None):
        """按 budgets 循环推进直到 completed；段间做 JSON 序列化往返。"""
        checkpoint = None
        paused = []
        for i in range(1000):
            budget = budgets[i % len(budgets)]
            response = self.svc.optimize_resumable(
                circuit, terms, values, config,
                noise=noise, step_budget=budget, checkpoint=checkpoint,
            )
            json.dumps(response)
            if response["status"] == "completed":
                return response, paused
            self.assertEqual(response["status"], "paused")
            paused.append(response)
            checkpoint = json.loads(json.dumps(response["checkpoint"]))
        self.fail("segmented run did not complete")

    def assert_same_result(self, circuit, terms, values, config, budgets, noise=None):
        expected = self.svc.optimize(circuit, terms, values, config, noise=noise)
        response, paused = self.run_segmented(
            circuit, terms, values, config, budgets, noise=noise,
        )
        self.assertIsNone(response["checkpoint"])
        self.assertEqual(response["result"], expected)
        # 暂停段的 progress 与最终 history 前缀一致。
        for segment in paused:
            progress = segment["progress"]
            iterations = progress["iterations"]
            self.assertEqual(progress["parameters"], expected["parameters"])
            self.assertEqual(progress["history"], expected["history"][:iterations + 1])
            last = progress["history"][-1]
            self.assertEqual(progress["values"], last["values"])
            self.assertEqual(progress["objective"], last["objective"])
            self.assertEqual(progress["gradient_norm"], last["gradient_norm"])
            self.assertIsNone(segment["result"])
        return expected

    def test_gradient_descent_various_budgets(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        for budgets in ((1,), (2,), (3, 1, 2), (7,), (1000,)):
            with self.subTest(budgets=budgets):
                self.assert_same_result(
                    rx_circuit(), terms, {"theta": 0.3}, gd_config(), budgets,
                )

    def test_adam_segmented(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        for budgets in ((1,), (4, 1), (1000,)):
            with self.subTest(budgets=budgets):
                self.assert_same_result(
                    rx_circuit(), terms, {"theta": 0.3}, adam_config(), budgets,
                )

    def test_multi_parameter_multi_term_with_noise(self):
        circuit = two_param_circuit()
        terms = [{"observable": "ZI", "coefficient": 2.0},
                 {"observable": "IZ", "coefficient": 0.5}]
        noise = {"single_qubit_depolarizing": 0.1, "two_qubit_depolarizing": 0.02}
        self.assert_same_result(
            circuit, terms, {"a": 0.2, "b": 0.9}, gd_config(), (1, 3), noise=noise,
        )
        self.assert_same_result(
            circuit, terms, {"a": 0.2, "b": 0.9}, adam_config(), (2,), noise=noise,
        )

    def test_max_iterations_boundary_completes_without_converging(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        config = gd_config(max_iterations=5, tolerance=0.0)
        expected = self.svc.optimize(rx_circuit(), terms, {"theta": 0.1}, config)
        self.assertFalse(expected["converged"])
        response, paused = self.run_segmented(
            rx_circuit(), terms, {"theta": 0.1}, config, (2,),
        )
        self.assertEqual(response["result"], expected)
        self.assertFalse(response["result"]["converged"])
        self.assertEqual(response["result"]["iterations"], 5)

    def test_initial_point_already_converged_completes_immediately(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        response = self.svc.optimize_resumable(
            rx_circuit(), terms, {"theta": math.pi},
            gd_config(tolerance=1e-6), step_budget=3,
        )
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["result"]["iterations"], 0)
        self.assertEqual(
            response["result"],
            self.svc.optimize(rx_circuit(), terms, {"theta": math.pi},
                              gd_config(tolerance=1e-6)),
        )

    def test_first_call_matches_iteration_zero(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        config = gd_config()
        expected = self.svc.optimize(rx_circuit(), terms, {"theta": 0.3}, config)
        response = self.svc.optimize_resumable(
            rx_circuit(), terms, {"theta": 0.3}, config, step_budget=1,
        )
        self.assertEqual(response["status"], "paused")
        progress = response["progress"]
        self.assertEqual(progress["iterations"], 1)
        self.assertEqual(progress["history"][:1], expected["history"][:1])
        self.assertEqual(progress["history"], expected["history"][:2])

    def test_checkpoint_is_json_native_and_inputs_not_mutated(self):
        circuit = rx_circuit()
        terms = [{"observable": "Z", "coefficient": 1.0}]
        values = {"theta": 0.3}
        config = gd_config()
        noise = {"single_qubit_depolarizing": 0.1}
        snapshot = (copy.deepcopy(circuit), copy.deepcopy(terms),
                    copy.deepcopy(values), copy.deepcopy(config), copy.deepcopy(noise))
        response = self.svc.optimize_resumable(
            circuit, terms, values, config, noise=noise, step_budget=2,
        )
        self.assertEqual((circuit, terms, values, config, noise), snapshot)
        checkpoint = response["checkpoint"]
        self.assertEqual(json.loads(json.dumps(checkpoint)), checkpoint)
        self.assertNotIn("-0.0", json.dumps(checkpoint))
        # 续算不修改检查点。
        before = copy.deepcopy(checkpoint)
        follow = self.svc.optimize_resumable(
            circuit, terms, values, config, noise=noise,
            step_budget=2, checkpoint=checkpoint,
        )
        self.assertEqual(checkpoint, before)
        json.dumps(follow)

    def test_cross_process_style_resume_from_serialized_checkpoint(self):
        terms = [{"observable": "Z", "coefficient": 1.0}]
        config = adam_config()
        first = self.svc.optimize_resumable(
            rx_circuit(), terms, {"theta": 0.3}, config, step_budget=2,
        )
        blob = json.dumps(first["checkpoint"], sort_keys=True)
        # 模拟跨进程：仅携带序列化文本，全新 Service 实例续算。
        other = Service()
        checkpoint = json.loads(blob)
        expected = self.svc.optimize(rx_circuit(), terms, {"theta": 0.3}, config)
        for _ in range(100):
            response = other.optimize_resumable(
                rx_circuit(), terms, {"theta": 0.3}, config,
                step_budget=5, checkpoint=checkpoint,
            )
            if response["status"] == "completed":
                self.assertEqual(response["result"], expected)
                return
            checkpoint = json.loads(json.dumps(response["checkpoint"]))
        self.fail("resumed run did not complete")


class ResumableBudgetValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.terms = [{"observable": "Z", "coefficient": 1.0}]

    def resumable(self, **overrides):
        args = {
            "circuit": rx_circuit(),
            "terms": self.terms,
            "values": {"theta": 0.3},
            "config": gd_config(),
            "step_budget": 1,
        }
        args.update(overrides)
        return self.svc.optimize_resumable(
            args["circuit"], args["terms"], args["values"], args["config"],
            noise=args.get("noise"), step_budget=args["step_budget"],
            checkpoint=args.get("checkpoint"),
        )

    def test_invalid_step_budget(self):
        for bad in (True, False, 0, -1, 1.5, "2", None, [1], math.nan):
            with self.subTest(bad=bad):
                e = state_err(self.resumable, step_budget=bad)
                self.assertEqual((e.code, e.path), ("invalid_step_budget", "step_budget"))

    def test_base_inputs_validated_before_budget(self):
        # 基础输入沿用原异常，且先于预算校验。
        with self.assertRaises(OptimizationError) as ctx:
            self.resumable(config=gd_config(learning_rate=-1), step_budget=0)
        self.assertEqual(ctx.exception.code, "invalid_optimizer_option")
        with self.assertRaises(OptimizationError) as ctx:
            self.resumable(terms=[], step_budget="x")
        self.assertEqual(ctx.exception.code, "invalid_terms")

    def test_budget_validated_before_checkpoint(self):
        e = state_err(self.resumable, step_budget=0, checkpoint="junk")
        self.assertEqual((e.code, e.path), ("invalid_step_budget", "step_budget"))

    def test_runtime_state_error_is_public(self):
        self.assertIs(PackageRuntimeStateError, RuntimeStateError)
        self.assertTrue(issubclass(RuntimeStateError, ValueError))


class ResumableCheckpointValidationTest(unittest.TestCase):
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
        args = {
            "circuit": self.circuit, "terms": self.terms, "values": self.values,
            "config": self.config, "noise": None,
        }
        args.update(overrides)
        return self.svc.optimize_resumable(
            args["circuit"], args["terms"], args["values"], args["config"],
            noise=args["noise"], step_budget=1, checkpoint=checkpoint,
        )

    def tampered(self, mutate):
        checkpoint = copy.deepcopy(self.checkpoint)
        mutate(checkpoint)
        return checkpoint

    def assert_invalid(self, checkpoint, code="invalid_checkpoint"):
        e = state_err(self.resume, checkpoint)
        self.assertEqual((e.code, e.path), (code, "checkpoint"))

    def test_not_an_object(self):
        for bad in ([], "checkpoint", 1, 1.0, True):
            with self.subTest(bad=bad):
                self.assert_invalid(bad)

    def test_unsupported_version(self):
        for bad in (0, 2, "1", 1.0, True, None):
            with self.subTest(bad=bad):
                self.assert_invalid(self.tampered(lambda c: c.__setitem__("version", bad)))
        self.assert_invalid(self.tampered(lambda c: c.pop("version")))

    def test_missing_and_unknown_fields(self):
        for field in ("fingerprint", "iterations", "values", "gradients",
                      "objective", "gradient_norm", "history", "optimizer"):
            with self.subTest(missing=field):
                self.assert_invalid(self.tampered(lambda c, f=field: c.pop(f)))
        self.assert_invalid(self.tampered(lambda c: c.__setitem__("extra", 1)))

    def test_wrong_types(self):
        cases = [
            lambda c: c.__setitem__("fingerprint", 1),
            lambda c: c.__setitem__("iterations", "2"),
            lambda c: c.__setitem__("iterations", 1.5),
            lambda c: c.__setitem__("iterations", -1),
            lambda c: c.__setitem__("iterations", True),
            lambda c: c.__setitem__("values", [0.3]),
            lambda c: c.__setitem__("values", {"theta": "0.3"}),
            lambda c: c.__setitem__("values", {"theta": True}),
            lambda c: c.__setitem__("gradients", None),
            lambda c: c.__setitem__("objective", "x"),
            lambda c: c.__setitem__("gradient_norm", [1]),
            lambda c: c.__setitem__("history", {"0": {}}),
            lambda c: c.__setitem__("history", ["nope", "nope", "nope"]),
            lambda c: c.__setitem__("optimizer", []),
            lambda c: c.__setitem__("optimizer", {"first_moment": {"theta": 0.0}}),
            lambda c: c.__setitem__("optimizer", {"first_moment": {"theta": 0.0},
                                                  "second_moment": {"theta": 0.0},
                                                  "third_moment": {}}),
        ]
        for mutate in cases:
            with self.subTest(mutate=mutate):
                self.assert_invalid(self.tampered(mutate))

    def test_non_finite_numbers(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                self.assert_invalid(self.tampered(
                    lambda c: c.__setitem__("values", {"theta": bad})))
            with self.subTest(bad=bad, field="objective"):
                self.assert_invalid(self.tampered(
                    lambda c: (c.__setitem__("objective", bad),
                               c["history"][-1].__setitem__("objective", bad))))
            with self.subTest(bad=bad, field="optimizer"):
                self.assert_invalid(self.tampered(
                    lambda c: c["optimizer"].__setitem__("first_moment", {"theta": bad})))
            with self.subTest(bad=bad, field="history"):
                self.assert_invalid(self.tampered(
                    lambda c: c["history"][0].__setitem__("gradient_norm", bad)))

    def test_history_structure(self):
        # 历史条目缺字段 / 多字段 / 类型错误。
        self.assert_invalid(self.tampered(
            lambda c: c["history"][0].pop("objective")))
        self.assert_invalid(self.tampered(
            lambda c: c["history"][0].__setitem__("extra", 1)))
        self.assert_invalid(self.tampered(
            lambda c: c["history"][0].__setitem__("iteration", "0")))

    def test_contradictory_state(self):
        # iterations 与 history 长度不符。
        self.assert_invalid(self.tampered(
            lambda c: c.__setitem__("iterations", c["iterations"] + 1)))
        # history 的 iteration 序列断裂。
        self.assert_invalid(self.tampered(
            lambda c: c["history"][1].__setitem__("iteration", 5)))
        # 当前状态与最后一条 history 不符。
        self.assert_invalid(self.tampered(
            lambda c: c.__setitem__("objective", c["objective"] + 1.0)))
        self.assert_invalid(self.tampered(
            lambda c: c["values"].__setitem__("theta", c["values"]["theta"] + 1.0)))
        # 参数集合与电路矛盾。
        self.assert_invalid(self.tampered(
            lambda c: (c.__setitem__("values", {"theta": 0.3, "extra": 1.0}),
                       c["history"][-1].__setitem__(
                           "values", {"theta": c["history"][-1]["values"]["theta"],
                                      "extra": 1.0}))))
        # 已达到 max_iterations 或已满足 tolerance 的矛盾状态。
        self.assert_invalid(self.tampered(
            lambda c: c.__setitem__("iterations", self.config["max_iterations"])))
        self.assert_invalid(self.tampered(
            lambda c: (c.__setitem__("gradient_norm", 0.0),
                       c["history"][-1].__setitem__("gradient_norm", 0.0))))

    def test_checkpoint_mismatch(self):
        mismatches = [
            {"values": {"theta": 0.4}},
            {"config": gd_config(learning_rate=0.4)},
            {"config": gd_config(max_iterations=51)},
            {"config": adam_config()},
            {"terms": [{"observable": "Z", "coefficient": 2.0}]},
            {"terms": [{"observable": "X", "coefficient": 1.0}]},
            {"circuit": {"qubit_count": 1, "parameters": ["theta"], "operations": [
                {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}}]}},
            {"noise": {"single_qubit_depolarizing": 0.1}},
        ]
        for overrides in mismatches:
            with self.subTest(overrides=overrides):
                e = state_err(self.resume, self.checkpoint, **overrides)
                self.assertEqual((e.code, e.path), ("checkpoint_mismatch", "checkpoint"))

    def test_valid_checkpoint_resumes(self):
        response = self.resume(self.checkpoint)
        self.assertEqual(response["status"], "paused")
        self.assertEqual(response["progress"]["iterations"],
                         self.checkpoint["iterations"] + 1)

    def test_serialized_checkpoint_round_trip_validates(self):
        blob = json.dumps(self.checkpoint)
        response = self.resume(json.loads(blob))
        self.assertEqual(response["status"], "paused")


if __name__ == "__main__":
    unittest.main()
