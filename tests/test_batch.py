import copy
import json
import math
import threading
import unittest
from unittest import mock

from qubitfabric import batch as batch_mod
from qubitfabric.service import (
    BatchExecutionError,
    CircuitValidationError,
    Service,
)


def batch_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except BatchExecutionError as exc:
        return exc
    raise AssertionError("BatchExecutionError not raised")


CIRCUIT = {
    "qubit_count": 2,
    "parameters": ["theta"],
    "operations": [
        {"gate": "h", "target": 0},
        {"gate": "cx", "control": 0, "target": 1},
        {"gate": "rx", "target": 1, "angle": {"parameter": "theta"}},
    ],
}


def job(job_id, observables=None, **extra):
    data = {"id": job_id, "observables": ["ZZ", "XX"] if observables is None else observables}
    data.update(extra)
    return data


class BatchSuccessTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.jobs = [
            job("a", values={"theta": 0.0}),
            job("b", values={"theta": math.pi / 2}, shots=100, seed=7),
            job("c", values={"theta": 1.0}, shots=50, seed=3,
                noise={"single_qubit_depolarizing": 0.1}),
        ]

    def test_results_shape_and_order(self):
        out = self.svc.batch_expectation(CIRCUIT, self.jobs)
        self.assertEqual(set(out), {"results", "summary"})
        self.assertEqual([item["id"] for item in out["results"]], ["a", "b", "c"])
        for item in out["results"]:
            self.assertEqual(item["status"], "succeeded")
            self.assertIsNone(item["error"])
            self.assertEqual(set(item), {"id", "status", "result", "error"})
            self.assertEqual(set(item["result"]), {"qubit_count", "shots", "results"})
        self.assertEqual(out["summary"], {"total": 3, "succeeded": 3, "failed": 0})

    def test_each_result_matches_single_expectation(self):
        out = self.svc.batch_expectation(CIRCUIT, self.jobs)
        for spec, item in zip(self.jobs, out["results"]):
            single = self.svc.expectation(
                CIRCUIT, spec["observables"],
                values=spec.get("values"), shots=spec.get("shots"),
                seed=spec.get("seed"), noise=spec.get("noise"),
            )
            self.assertEqual(item["result"], single)

    def test_sampled_counts_and_seed_reproduced(self):
        jobs = [job("s1", ["ZI", "IZ"], values={"theta": 0.3}, shots=200, seed=42)]
        first = self.svc.batch_expectation(CIRCUIT, jobs)
        second = self.svc.batch_expectation(CIRCUIT, copy.deepcopy(jobs))
        self.assertEqual(first, second)
        single = self.svc.expectation(CIRCUIT, ["ZI", "IZ"],
                                      values={"theta": 0.3}, shots=200, seed=42)
        self.assertEqual(first["results"][0]["result"], single)
        # 默认 seed 与单次入口省略 seed 一致（采样时为 0）。
        default_seed = [job("d", ["ZZ"], values={"theta": 0.0}, shots=40)]
        batch_result = self.svc.batch_expectation(CIRCUIT, default_seed)["results"][0]["result"]
        single_result = self.svc.expectation(CIRCUIT, ["ZZ"],
                                             values={"theta": 0.0}, shots=40)
        self.assertEqual(batch_result, single_result)

    def test_results_identical_across_concurrency_levels(self):
        baseline = self.svc.batch_expectation(CIRCUIT, self.jobs, max_concurrency=1)
        for level in (2, 3, 8):
            other = self.svc.batch_expectation(CIRCUIT, self.jobs, max_concurrency=level)
            self.assertEqual(other, baseline)

    def test_default_concurrency_is_one(self):
        # 省略 max_concurrency 与显式传 1 等价。
        omitted = self.svc.batch_expectation(CIRCUIT, self.jobs)
        explicit = self.svc.batch_expectation(CIRCUIT, self.jobs, max_concurrency=1)
        self.assertEqual(omitted, explicit)

    def test_output_is_json_native(self):
        out = self.svc.batch_expectation(CIRCUIT, self.jobs, max_concurrency=2)
        json.dumps(out)  # 不抛异常即全部为 JSON 原生类型

    def test_inputs_not_modified(self):
        circuit_copy = copy.deepcopy(CIRCUIT)
        jobs_copy = copy.deepcopy(self.jobs)
        self.svc.batch_expectation(CIRCUIT, self.jobs, max_concurrency=3)
        self.assertEqual(CIRCUIT, circuit_copy)
        self.assertEqual(self.jobs, jobs_copy)


class ConcurrencyBoundTest(unittest.TestCase):
    def test_never_exceeds_max_concurrency(self):
        svc = Service()
        jobs = [job(f"j{i}", values={"theta": 0.1 * i}) for i in range(4)]
        active = 0
        lock = threading.Lock()
        seen_max = 0
        pair = threading.Barrier(2)
        original = batch_mod.estimate_expectation

        def tracked(circuit, observables, **kwargs):
            nonlocal active, seen_max
            with lock:
                active += 1
                seen_max = max(seen_max, active)
            try:
                pair.wait(timeout=2.0)  # 并行度为 1 时这里必然超时
            finally:
                with lock:
                    active -= 1
            return original(circuit, observables, **kwargs)

        with mock.patch.object(batch_mod, "estimate_expectation", side_effect=tracked):
            out = svc.batch_expectation(CIRCUIT, jobs, max_concurrency=2)
        self.assertEqual(seen_max, 2)
        self.assertEqual(out["summary"], {"total": 4, "succeeded": 4, "failed": 0})

    def test_concurrency_one_executes_serially(self):
        svc = Service()
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(3)]
        active = 0
        seen_max = 0
        lock = threading.Lock()
        original = batch_mod.estimate_expectation

        def tracked(circuit, observables, **kwargs):
            nonlocal active, seen_max
            with lock:
                active += 1
                seen_max = max(seen_max, active)
                active -= 1
            return original(circuit, observables, **kwargs)

        with mock.patch.object(batch_mod, "estimate_expectation", side_effect=tracked):
            svc.batch_expectation(CIRCUIT, jobs, max_concurrency=1)
        self.assertEqual(seen_max, 1)


class BatchFailureIsolationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_binding_and_simulation_failures_only_fail_that_job(self):
        jobs = [
            job("ok", values={"theta": 0.0}),
            job("missing-binding"),  # 缺参数绑定
            job("bad-observable", ["ZZZ"], values={"theta": 0.0}),  # 长度不符
            job("bad-shots", values={"theta": 0.0}, shots=0),
            job("ok2", values={"theta": 1.0}, shots=10, seed=1),
        ]
        out = self.svc.batch_expectation(CIRCUIT, jobs, max_concurrency=3)
        results = out["results"]
        self.assertEqual([item["id"] for item in results],
                         ["ok", "missing-binding", "bad-observable", "bad-shots", "ok2"])
        statuses = [item["status"] for item in results]
        self.assertEqual(statuses, ["succeeded", "failed", "failed", "failed", "succeeded"])
        self.assertEqual(out["summary"], {"total": 5, "succeeded": 2, "failed": 3})

        failed = results[1]
        self.assertIsNone(failed["result"])
        self.assertEqual(failed["error"]["type"], "ParameterBindingError")
        self.assertEqual(failed["error"]["code"], "missing_parameter")
        self.assertEqual(failed["error"]["path"], "theta")

        self.assertEqual(results[2]["error"]["type"], "SimulationError")
        self.assertEqual(results[2]["error"]["code"], "invalid_observable")
        self.assertEqual(results[2]["error"]["path"], "observables[0]")

        self.assertEqual(results[3]["error"]["type"], "SimulationError")
        self.assertEqual(results[3]["error"]["code"], "invalid_shots")
        self.assertEqual(results[3]["error"]["path"], "shots")

        for failed_item in results[1:4]:
            self.assertEqual(set(failed_item), {"id", "status", "result", "error"})
            self.assertEqual(
                set(failed_item["error"]), {"type", "code", "path", "message"},
            )

        self.assertEqual(
            results[0]["result"],
            self.svc.expectation(CIRCUIT, ["ZZ", "XX"], values={"theta": 0.0}),
        )
        self.assertEqual(
            results[4]["result"],
            self.svc.expectation(CIRCUIT, ["ZZ", "XX"],
                                 values={"theta": 1.0}, shots=10, seed=1),
        )

    def test_failures_identical_across_concurrency_levels(self):
        jobs = [
            job("ok", values={"theta": 0.0}),
            job("broken"),
            job("ok2", values={"theta": 0.5}),
        ]
        one = self.svc.batch_expectation(CIRCUIT, jobs, max_concurrency=1)
        many = self.svc.batch_expectation(CIRCUIT, jobs, max_concurrency=4)
        self.assertEqual(many, one)

    def test_state_space_limit_fails_job(self):
        circuit = {"qubit_count": 21}
        out = self.svc.batch_expectation(circuit, [job("big", ["I" * 21])])
        (only,) = out["results"]
        self.assertEqual(only["status"], "failed")
        self.assertEqual(only["error"]["type"], "SimulationError")
        self.assertEqual(only["error"]["code"], "state_space_too_large")
        self.assertEqual(only["error"]["path"], "qubit_count")
        self.assertEqual(out["summary"], {"total": 1, "succeeded": 0, "failed": 1})


class BatchRequestValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_invalid_circuit_raises_circuit_error_before_jobs(self):
        with self.assertRaises(CircuitValidationError) as ctx:
            self.svc.batch_expectation({"qubit_count": -1}, [])
        self.assertEqual(ctx.exception.code, "invalid_value")
        # jobs 结构再差也不改变电路错误的优先级。
        with self.assertRaises(CircuitValidationError):
            self.svc.batch_expectation("nope", None)

    def test_jobs_must_be_non_empty_array(self):
        for bad in (None, [], {}, "jobs", 1, True):
            exc = batch_err(self.svc.batch_expectation, CIRCUIT, bad)
            self.assertEqual((exc.code, exc.path), ("invalid_batch", "jobs"), bad)

    def test_job_must_be_object(self):
        exc = batch_err(self.svc.batch_expectation, CIRCUIT, [job("ok", values={"theta": 0}), "x"])
        self.assertEqual((exc.code, exc.path), ("invalid_job", "jobs[1]"))

    def test_id_required_non_empty_string(self):
        cases = [
            ([{"observables": ["Z"]}], "jobs[0].id"),
            ([{"id": 4, "observables": ["Z"]}], "jobs[0].id"),
            ([{"id": "", "observables": ["Z"]}], "jobs[0].id"),
            ([{"id": None, "observables": ["Z"]}], "jobs[0].id"),
        ]
        for bad_jobs, path in cases:
            exc = batch_err(self.svc.batch_expectation, {"qubit_count": 1}, bad_jobs)
            self.assertEqual((exc.code, exc.path), ("invalid_job", path), bad_jobs)

    def test_observables_required(self):
        exc = batch_err(self.svc.batch_expectation, {"qubit_count": 1}, [{"id": "a"}])
        self.assertEqual((exc.code, exc.path), ("invalid_job", "jobs[0].observables"))

    def test_unknown_job_field_rejected(self):
        exc = batch_err(
            self.svc.batch_expectation, {"qubit_count": 1},
            [{"id": "a", "observables": ["Z"], "bogus": 1}],
        )
        self.assertEqual((exc.code, exc.path), ("invalid_job", "jobs[0].bogus"))

    def test_duplicate_id_points_to_later_occurrence(self):
        jobs = [
            {"id": "x", "observables": ["Z"]},
            {"id": "y", "observables": ["Z"]},
            {"id": "x", "observables": ["Z"]},
        ]
        exc = batch_err(self.svc.batch_expectation, {"qubit_count": 1}, jobs)
        self.assertEqual((exc.code, exc.path), ("duplicate_job_id", "jobs[2].id"))

    def test_invalid_concurrency(self):
        jobs = [{"id": "a", "observables": ["Z"]}]
        for bad in (0, -1, 1.5, "2", True, False, [], {}):
            exc = batch_err(self.svc.batch_expectation, {"qubit_count": 1}, jobs, bad)
            self.assertEqual(
                (exc.code, exc.path), ("invalid_concurrency", "max_concurrency"), bad,
            )

    def test_concurrency_validated_before_jobs_structure(self):
        exc = batch_err(self.svc.batch_expectation, CIRCUIT, [], 0)
        self.assertEqual(exc.code, "invalid_concurrency")

    def test_no_jobs_started_when_request_invalid(self):
        # 请求级错误发生在任何作业执行之前。
        with mock.patch.object(batch_mod, "estimate_expectation") as called:
            batch_err(self.svc.batch_expectation, CIRCUIT, [{"id": "a"}], 0)
            called.assert_not_called()


class BatchErrorHierarchyTest(unittest.TestCase):
    def test_batch_error_is_value_error_with_code_and_path(self):
        exc = BatchExecutionError("invalid_batch", "jobs")
        self.assertIsInstance(exc, ValueError)
        self.assertEqual(exc.code, "invalid_batch")
        self.assertEqual(exc.path, "jobs")

    def test_exported_from_package_and_service(self):
        from qubitfabric import BatchExecutionError as PkgError
        from qubitfabric.service import BatchExecutionError as SvcError
        self.assertIs(PkgError, SvcError)


if __name__ == "__main__":
    unittest.main()
