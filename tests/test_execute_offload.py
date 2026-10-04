"""Service.execute_batch_offload 的契约测试。

执行入口必须：
- 请求级校验与异常顺序沿用 plan_batch_offload（电路 → backends →
  jobs 结构），其后校验 max_concurrency 与 executors，请求级失败时
  不调用任何执行器、不返回部分结果；
- plan 与同输入 plan_batch_offload 的结果逐值一致；results 与 jobs
  同序；rejected / no_eligible_backend 项保留规划详情且不执行；
- assigned 作业只调用所选后端执行器一次，入参为规范化电路与作业的
  独立副本；合法返回记 succeeded（深拷贝 result），非法形态记
  failed（invalid_backend_result），抛异常记 failed
  （backend_failure，保留原异常类型名与消息），不重试、不连坐；
- 全局并行数不超过 max_concurrency，单后端并行数不超过 slots，
  完成先后不影响结果顺序，不同并发度输出一致；
- summary 五类计数之和等于 jobs 数；输出仅含 JSON 原生类型；
- 不修改输入或执行器返回对象。
"""

import copy
import json
import threading
import unittest

from qubitfabric import OffloadExecutionError as PkgOffloadExecutionError
from qubitfabric.service import (
    BatchExecutionError,
    CircuitValidationError,
    OffloadExecutionError,
    OffloadPlanningError,
    Service,
)

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


def backend(backend_id="b0", slots=4, reps=None, **limits):
    data = {
        "id": backend_id,
        "representations": ["state_vector", "density_matrix"] if reps is None else reps,
        "max_state_bytes": limits.pop("max_state_bytes", 10 ** 12),
        "max_circuit_evaluations": limits.pop("max_circuit_evaluations", 10 ** 6),
        "max_gate_applications": limits.pop("max_gate_applications", 10 ** 6),
        "max_total_shots": limits.pop("max_total_shots", 10 ** 9),
        "slots": slots,
    }
    assert not limits
    return data


def local_executor(svc):
    """用本地 expectation 实现充当后端执行器。"""
    def execute(circuit, spec):
        return svc.expectation(
            circuit, spec["observables"],
            values=spec.get("values"), shots=spec.get("shots"),
            seed=spec.get("seed"), noise=spec.get("noise"),
        )
    return execute


def execution_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except OffloadExecutionError as exc:
        return exc
    raise AssertionError("OffloadExecutionError not raised")


class ExecuteSuccessTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("a", slots=4), backend("b", slots=4)]
        self.jobs = [
            job("exact", values={"theta": 0.0}),
            job("sampled", values={"theta": 0.5}, shots=40, seed=3),
            job("rejected"),
            job("homeless", ["ZZ"], values={"theta": 0.0}),
        ]
        # 填满 a/b 的槽位，使最后一个作业无后端可去（slots 调小）。
        self.backends = [backend("a", slots=2), backend("b", slots=1)]
        self.jobs = [
            job("a1", values={"theta": 0.0}),
            job("b1", values={"theta": 0.0}),
            job("a2", values={"theta": 0.0}),
            job("homeless", ["ZZ"], values={"theta": 0.0}),
            job("rejected"),
        ]
        self.executors = {"a": local_executor(self.svc), "b": local_executor(self.svc)}

    def test_response_shape_and_plan_identical_to_plan_endpoint(self):
        out = self.svc.execute_batch_offload(CIRCUIT, self.jobs, self.backends, self.executors)
        self.assertEqual(set(out), {"plan", "results", "summary"})
        self.assertEqual(
            out["plan"],
            self.svc.plan_batch_offload(CIRCUIT, self.jobs, self.backends),
        )
        self.assertEqual(
            [item["id"] for item in out["results"]],
            ["a1", "b1", "a2", "homeless", "rejected"],
        )

    def test_succeeded_items_match_local_expectation(self):
        out = self.svc.execute_batch_offload(CIRCUIT, self.jobs, self.backends, self.executors)
        items = out["results"]
        for index in (0, 1, 2):
            spec = self.jobs[index]
            item = items[index]
            self.assertEqual(item["status"], "succeeded")
            self.assertIsNone(item["error"])
            self.assertEqual(set(item), {"id", "status", "backend_id", "result", "error"})
            self.assertEqual(
                item["result"],
                self.svc.expectation(
                    CIRCUIT, spec["observables"],
                    values=spec.get("values"), shots=spec.get("shots"),
                    seed=spec.get("seed"), noise=spec.get("noise"),
                ),
            )
        self.assertEqual([items[i]["backend_id"] for i in (0, 1, 2)], ["a", "b", "a"])

    def test_assigned_job_calls_selected_backend_exactly_once(self):
        calls = []

        def make(bid):
            def execute(circuit, spec):
                calls.append((bid, spec["id"]))
                return local_executor(self.svc)(circuit, spec)
            return execute

        executors = {"a": make("a"), "b": make("b")}
        self.svc.execute_batch_offload(CIRCUIT, self.jobs, self.backends, executors)
        self.assertEqual(sorted(calls), sorted([("a", "a1"), ("b", "b1"), ("a", "a2")]))

    def test_rejected_and_no_eligible_items_keep_planning_details(self):
        out = self.svc.execute_batch_offload(CIRCUIT, self.jobs, self.backends, self.executors)
        homeless = out["results"][3]
        self.assertEqual(homeless["status"], "no_eligible_backend")
        self.assertIsNone(homeless["result"])
        self.assertIsNone(homeless["error"])
        self.assertIsNone(homeless["requirements"])
        self.assertEqual(homeless["reasons"], [["no_slot"], ["no_slot"]])

        rejected = out["results"][4]
        self.assertEqual(rejected["status"], "rejected")
        self.assertIsNone(rejected["result"])
        self.assertIsNone(rejected["error"])
        self.assertIsNone(rejected["requirements"])
        self.assertEqual(
            rejected["validation_error"]["type"], "ParameterBindingError",
        )
        self.assertEqual(rejected["validation_error"]["code"], "missing_parameter")

    def test_summary_counts_and_partition(self):
        out = self.svc.execute_batch_offload(CIRCUIT, self.jobs, self.backends, self.executors)
        self.assertEqual(
            out["summary"],
            {"total": 5, "succeeded": 3, "failed": 0,
             "rejected": 1, "no_eligible_backend": 1},
        )
        partition = (
            out["summary"]["succeeded"] + out["summary"]["failed"]
            + out["summary"]["rejected"] + out["summary"]["no_eligible_backend"]
        )
        self.assertEqual(partition, out["summary"]["total"])

    def test_output_is_json_native(self):
        out = self.svc.execute_batch_offload(CIRCUIT, self.jobs, self.backends, self.executors)
        json.dumps(out)


class ExecutorContractTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("a")]

    def test_executor_receives_normalized_circuit_and_job_copy(self):
        seen = {}

        def execute(circuit, spec):
            seen["circuit"] = circuit
            seen["job"] = spec
            return local_executor(self.svc)(circuit, spec)

        jobs = [job("j", values={"theta": 0.0})]
        self.svc.execute_batch_offload(CIRCUIT, jobs, self.backends, {"a": execute})
        self.assertEqual(seen["circuit"], self.svc.create_circuit(CIRCUIT))
        self.assertEqual(seen["job"], jobs[0])
        self.assertIsNot(seen["circuit"], CIRCUIT)
        self.assertIsNot(seen["job"], jobs[0])

    def test_inputs_not_modified_even_if_executor_mutates_arguments(self):
        def execute(circuit, spec):
            circuit["operations"].append({"gate": "x", "target": 0})
            circuit["qubit_count"] = 99
            spec["id"] = "mutated"
            spec["observables"] = ["II"]
            return {"qubit_count": 2, "shots": None,
                    "results": [{"observable": "ZZ", "expectation": 1.0},
                                {"observable": "XX", "expectation": 0.0}]}

        circuit_copy = copy.deepcopy(CIRCUIT)
        jobs = [job("j", values={"theta": 0.0})]
        jobs_copy = copy.deepcopy(jobs)
        backends_copy = copy.deepcopy(self.backends)
        out = self.svc.execute_batch_offload(CIRCUIT, jobs, self.backends, {"a": execute})
        self.assertEqual(CIRCUIT, circuit_copy)
        self.assertEqual(jobs, jobs_copy)
        self.assertEqual(self.backends, backends_copy)
        self.assertEqual(out["results"][0]["id"], "j")

    def test_backend_return_object_is_deep_copied_into_output(self):
        returned = {
            "qubit_count": 2, "shots": None,
            "results": [{"observable": "ZZ", "expectation": 1.0},
                        {"observable": "XX", "expectation": 0.0}],
        }
        returned_copy = copy.deepcopy(returned)

        def execute(circuit, spec):
            return returned

        jobs = [job("j", values={"theta": 0.0})]
        out = self.svc.execute_batch_offload(CIRCUIT, jobs, self.backends, {"a": execute})
        self.assertEqual(out["results"][0]["result"], returned_copy)
        self.assertIsNot(out["results"][0]["result"], returned)
        out["results"][0]["result"]["results"][0]["expectation"] = 999
        self.assertEqual(returned, returned_copy)


class InvalidBackendResultTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("a")]
        self.jobs = [job("j", values={"theta": 0.0})]

    def run_with_result(self, raw):
        out = self.svc.execute_batch_offload(
            CIRCUIT, self.jobs, self.backends, {"a": lambda c, s: raw},
        )
        return out["results"][0]

    def test_non_object_results_fail(self):
        for raw in (None, 5, "x", [], True):
            item = self.run_with_result(raw)
            self.assertEqual(item["status"], "failed", raw)
            self.assertIsNone(item["result"])
            self.assertEqual(item["backend_id"], "a")
            self.assertEqual(
                item["error"],
                {"type": "OffloadExecutionError", "code": "invalid_backend_result",
                 "message": "backend result must match the expectation result shape"},
            )

    def test_wrong_top_level_fields_fail(self):
        for raw in (
            {},
            {"qubit_count": 2, "shots": None},
            {"qubit_count": 2, "shots": None,
             "results": [{"observable": "ZZ", "expectation": 1.0},
                         {"observable": "XX", "expectation": 0.0}], "extra": 1},
        ):
            self.assertEqual(self.run_with_result(raw)["error"]["code"],
                             "invalid_backend_result", raw)

    def test_qubit_count_must_be_non_negative_int_excluding_bool(self):
        good = [{"observable": "ZZ", "expectation": 1.0},
                {"observable": "XX", "expectation": 0.0}]
        for qc in (True, False, -1, 1.5, "2", None):
            raw = {"qubit_count": qc, "shots": None, "results": good}
            self.assertEqual(self.run_with_result(raw)["error"]["code"],
                             "invalid_backend_result", qc)

    def test_shots_must_be_null_or_positive_int(self):
        entry = [{"observable": "ZZ", "expectation": 1.0},
                 {"observable": "XX", "expectation": 0.0}]
        for shots in (0, -1, 1.5, "3", True, False):
            raw = {"qubit_count": 2, "shots": shots, "results": entry}
            self.assertEqual(self.run_with_result(raw)["error"]["code"],
                             "invalid_backend_result", shots)

    def test_results_length_must_match_observables(self):
        base = {"qubit_count": 2, "shots": None}
        bad_lengths = [
            [],
            [{"observable": "ZZ", "expectation": 1.0}],
            [{"observable": "ZZ", "expectation": 1.0},
             {"observable": "XX", "expectation": 0.0},
             {"observable": "ZI", "expectation": 0.0}],
        ]
        for entries in bad_lengths:
            raw = {**base, "results": entries}
            self.assertEqual(self.run_with_result(raw)["error"]["code"],
                             "invalid_backend_result", entries)

    def test_exact_entries_must_have_observable_and_expectation_only(self):
        good_expectation = 1.0
        cases = [
            [{"observable": "ZZ"}],  # 缺 expectation
            [{"observable": "ZZ", "expectation": good_expectation},
             {"observable": "XX", "expectation": 0.0, "counts": {"positive": 1, "negative": 0}}],
            [{"observable": "ZZ", "expectation": "x"},
             {"observable": "XX", "expectation": 0.0}],
            [{"observable": "ZZ", "expectation": True},
             {"observable": "XX", "expectation": 0.0}],
            [{"observable": "Z", "expectation": 1.0},
             {"observable": "XX", "expectation": 0.0}],
            [{"observable": "QQ", "expectation": 1.0},
             {"observable": "XX", "expectation": 0.0}],
        ]
        for entries in cases:
            raw = {"qubit_count": 2, "shots": None, "results": entries}
            self.assertEqual(self.run_with_result(raw)["error"]["code"],
                             "invalid_backend_result", entries)

    def test_sampled_entries_require_counts_consistent_with_shots(self):
        def entry(positive, negative, obs="ZZ"):
            return {"observable": obs, "expectation": 0.0,
                    "counts": {"positive": positive, "negative": negative}}

        cases = [
            {"qubit_count": 2, "shots": 10,
             "results": [entry(5, 5), {"observable": "XX", "expectation": 0.0}]},
            {"qubit_count": 2, "shots": 10,
             "results": [entry(5, 5), entry(4, 4)]},
            {"qubit_count": 2, "shots": 10,
             "results": [entry(5, 6), entry(5, 5)]},
            {"qubit_count": 2, "shots": 10,
             "results": [entry(-1, 11), entry(5, 5)]},
            {"qubit_count": 2, "shots": 10,
             "results": [entry(True, 10), entry(5, 5)]},
            {"qubit_count": 2, "shots": 10,
             "results": [entry(5, 5, "XX1"), entry(5, 5)]},
        ]
        for raw in cases:
            self.assertEqual(self.run_with_result(raw)["error"]["code"],
                             "invalid_backend_result", raw)

    def test_non_json_native_values_fail(self):
        raw = {"qubit_count": 2, "shots": None,
               "results": [{"observable": "ZZ", "expectation": 1.0},
                           {"observable": "XX", "expectation": float("nan")}]}
        # NaN 被 Python json 接受但不是合法 JSON 值；形态校验仍视为失败
        # （期望 JSON 对象语义）。
        item = self.run_with_result(raw)
        self.assertEqual(item["status"], "failed")

    def test_qubit_count_must_match_normalized_circuit(self):
        entries = [{"observable": "ZZ", "expectation": 1.0},
                   {"observable": "XX", "expectation": 0.0}]
        for qc in (1, 3):
            raw = {"qubit_count": qc, "shots": None, "results": entries}
            self.assertEqual(self.run_with_result(raw)["error"]["code"],
                             "invalid_backend_result", qc)

    def test_shots_must_match_job_shots(self):
        entries_exact = [{"observable": "ZZ", "expectation": 1.0},
                         {"observable": "XX", "expectation": 0.0}]
        # 精确作业却返回采样形态。
        raw = {"qubit_count": 2, "shots": 10, "results": [
            {"observable": "ZZ", "expectation": 0.0, "counts": {"positive": 5, "negative": 5}},
            {"observable": "XX", "expectation": 0.0, "counts": {"positive": 5, "negative": 5}},
        ]}
        self.assertEqual(self.run_with_result(raw)["error"]["code"], "invalid_backend_result")

        # 采样作业却返回精确形态（或 shots 数不一致）。
        sampled_job = job("s", values={"theta": 0.0}, shots=20, seed=1)
        for raw in (
            {"qubit_count": 2, "shots": None, "results": entries_exact},
            {"qubit_count": 2, "shots": 10, "results": [
                {"observable": "ZZ", "expectation": 0.0, "counts": {"positive": 5, "negative": 5}},
                {"observable": "XX", "expectation": 0.0, "counts": {"positive": 5, "negative": 5}},
            ]},
        ):
            out = self.svc.execute_batch_offload(
                CIRCUIT, [sampled_job], self.backends, {"a": lambda c, s: raw},
            )
            self.assertEqual(out["results"][0]["error"]["code"],
                             "invalid_backend_result", raw)

    def test_failure_is_isolated_and_counted(self):
        jobs = [
            job("good", values={"theta": 0.0}),
            job("bad", values={"theta": 0.0}),
            job("good2", values={"theta": 1.0}),
        ]

        def execute(circuit, spec):
            if spec["id"] == "bad":
                return {"not": "an expectation result"}
            return local_executor(self.svc)(circuit, spec)

        out = self.svc.execute_batch_offload(CIRCUIT, jobs, self.backends, {"a": execute})
        statuses = [item["status"] for item in out["results"]]
        self.assertEqual(statuses, ["succeeded", "failed", "succeeded"])
        self.assertEqual(
            out["summary"],
            {"total": 3, "succeeded": 2, "failed": 1,
             "rejected": 0, "no_eligible_backend": 0},
        )


class BackendFailureTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("a")]

    def test_executor_exception_fails_only_that_job_with_original_type(self):
        class CustomBackendError(RuntimeError):
            pass

        def execute(circuit, spec):
            if spec["id"] == "boom":
                raise CustomBackendError("backend exploded")
            return local_executor(self.svc)(circuit, spec)

        jobs = [job("ok", values={"theta": 0.0}), job("boom", values={"theta": 0.0}),
                job("ok2", values={"theta": 1.0})]
        out = self.svc.execute_batch_offload(CIRCUIT, jobs, self.backends, {"a": execute})
        self.assertEqual([item["status"] for item in out["results"]],
                         ["succeeded", "failed", "succeeded"])
        failed = out["results"][1]
        self.assertIsNone(failed["result"])
        self.assertEqual(failed["backend_id"], "a")
        self.assertEqual(
            failed["error"],
            {"type": "CustomBackendError", "code": "backend_failure",
             "message": "backend exploded"},
        )
        self.assertEqual(set(failed["error"]), {"type", "code", "message"})

    def test_no_retry_after_exception(self):
        attempts = []

        def execute(circuit, spec):
            attempts.append(spec["id"])
            raise RuntimeError("once only")

        self.svc.execute_batch_offload(
            CIRCUIT, [job("j", values={"theta": 0.0})], self.backends, {"a": execute},
        )
        self.assertEqual(attempts, ["j"])

    def test_base_exception_subclasses_also_fail_item(self):
        def execute(circuit, spec):
            raise RuntimeError("plain")

        out = self.svc.execute_batch_offload(
            CIRCUIT, [job("j", values={"theta": 0.0})], self.backends, {"a": execute},
        )
        self.assertEqual(out["results"][0]["error"]["type"], "RuntimeError")


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_never_exceeds_global_max_concurrency(self):
        backends = [backend("a", slots=8)]
        jobs = [job(f"j{i}", values={"theta": 0.1 * i}) for i in range(4)]
        active = 0
        peak = 0
        lock = threading.Lock()
        pair = threading.Barrier(2)

        def execute(circuit, spec):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                pair.wait(timeout=2.0)
            finally:
                with lock:
                    active -= 1
            return local_executor(self.svc)(circuit, spec)

        out = self.svc.execute_batch_offload(
            CIRCUIT, jobs, backends, {"a": execute}, max_concurrency=2,
        )
        self.assertEqual(peak, 2)
        self.assertEqual(out["summary"]["succeeded"], 4)

    def test_never_exceeds_per_backend_slots(self):
        backends = [backend("a", slots=1), backend("b", slots=2)]
        # 三个分配：a 一个、b 两个；第四个无槽位不执行。
        jobs = [job(f"j{i}", ["ZZ"], values={"theta": 0.1 * i}) for i in range(4)]
        active = {"a": 0, "b": 0}
        peak = {"a": 0, "b": 0}
        lock = threading.Lock()

        def make(bid):
            def execute(circuit, spec):
                with lock:
                    active[bid] += 1
                    peak[bid] = max(peak[bid], active[bid])
                barrier.wait(timeout=2.0)
                with lock:
                    active[bid] -= 1
                return {"qubit_count": 2, "shots": None,
                        "results": [{"observable": "ZZ", "expectation": 0.0}]}
            return execute

        barrier = threading.Barrier(3)
        out = self.svc.execute_batch_offload(
            CIRCUIT, jobs, backends,
            {"a": make("a"), "b": make("b")}, max_concurrency=8,
        )
        self.assertEqual(peak, {"a": 1, "b": 2})
        self.assertEqual(out["summary"]["succeeded"], 3)
        self.assertEqual(out["summary"]["no_eligible_backend"], 1)

    def test_completion_order_does_not_change_result_order(self):
        backends = [backend("a", slots=8)]
        jobs = [job(f"j{i}", values={"theta": 0.1 * i}) for i in range(6)]
        delays = {"j0": 0.3, "j1": 0.0, "j2": 0.2, "j3": 0.05, "j4": 0.15, "j5": 0.1}

        def execute(circuit, spec):
            event.wait(delays[spec["id"]])
            return local_executor(self.svc)(circuit, spec)

        event = threading.Event()
        out = self.svc.execute_batch_offload(
            CIRCUIT, jobs, backends, {"a": execute}, max_concurrency=6,
        )
        self.assertEqual([item["id"] for item in out["results"]],
                         [f"j{i}" for i in range(6)])

    def test_results_identical_across_concurrency_levels(self):
        backends = [backend("a", slots=8)]
        jobs = [
            job("a", values={"theta": 0.0}),
            job("b", values={"theta": 0.5}, shots=40, seed=3),
            job("c", values={"theta": 1.0},
                noise={"single_qubit_depolarizing": 0.1}),
        ]
        executors = {"a": local_executor(self.svc)}
        baseline = self.svc.execute_batch_offload(
            CIRCUIT, jobs, backends, executors, max_concurrency=1,
        )
        for level in (2, 3, 8):
            other = self.svc.execute_batch_offload(
                CIRCUIT, jobs, backends,
                {"a": local_executor(self.svc)}, max_concurrency=level,
            )
            self.assertEqual(other, baseline)

    def test_default_concurrency_is_one(self):
        backends = [backend("a", slots=8)]
        jobs = [job("j", values={"theta": 0.0})]
        executors = {"a": local_executor(self.svc)}
        omitted = self.svc.execute_batch_offload(CIRCUIT, jobs, backends, executors)
        explicit = self.svc.execute_batch_offload(
            CIRCUIT, jobs, backends, {"a": local_executor(self.svc)}, max_concurrency=1,
        )
        self.assertEqual(omitted, explicit)


class RequestValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("a")]
        self.jobs = [job("j", values={"theta": 0.0})]

    @staticmethod
    def good_result(circuit, spec):
        return {"qubit_count": 2, "shots": None,
                "results": [{"observable": "ZZ", "expectation": 1.0},
                            {"observable": "XX", "expectation": 0.0}]}

    def test_circuit_error_has_priority_over_everything(self):
        with self.assertRaises(CircuitValidationError) as ctx:
            self.svc.execute_batch_offload({"qubit_count": -1}, None, None, None, 0)
        self.assertEqual(ctx.exception.code, "invalid_value")

    def test_backends_error_before_jobs_concurrency_and_executors(self):
        with self.assertRaises(OffloadPlanningError) as ctx:
            self.svc.execute_batch_offload(CIRCUIT, None, None, None, 0)
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("invalid_backends", "backends"))

    def test_jobs_structure_error_before_executor_validation(self):
        with self.assertRaises(BatchExecutionError) as ctx:
            self.svc.execute_batch_offload(CIRCUIT, [], self.backends, "not-a-map")
        self.assertEqual((ctx.exception.code, ctx.exception.path), ("invalid_batch", "jobs"))

    def test_concurrency_validated_before_executors(self):
        with self.assertRaises(BatchExecutionError) as ctx:
            self.svc.execute_batch_offload(CIRCUIT, self.jobs, self.backends, "not-a-map", 0)
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("invalid_concurrency", "max_concurrency"))

    def test_invalid_concurrency_forms(self):
        for bad in (0, -1, 1.5, "2", True, False, [], {}):
            with self.assertRaises(BatchExecutionError) as ctx:
                self.svc.execute_batch_offload(
                    CIRCUIT, self.jobs, self.backends, {"a": self.good_result}, bad,
                )
            self.assertEqual(
                (ctx.exception.code, ctx.exception.path),
                ("invalid_concurrency", "max_concurrency"),
                bad,
            )

    def test_executors_must_be_a_mapping(self):
        for bad in (None, [], "executors", 1, True):
            exc = execution_err(
                self.svc.execute_batch_offload,
                CIRCUIT, self.jobs, self.backends, bad,
            )
            self.assertEqual((exc.code, exc.path), ("invalid_executors", "executors"), bad)

    def test_executors_keys_must_be_strings(self):
        class WeirdDict(dict):
            pass

        executors = WeirdDict()
        executors[0] = self.good_result
        exc = execution_err(
            self.svc.execute_batch_offload,
            CIRCUIT, self.jobs, self.backends, executors,
        )
        self.assertEqual((exc.code, exc.path), ("invalid_executors", "executors"))

    def test_executors_keys_must_cover_backend_ids_exactly(self):
        # 缺键
        exc = execution_err(
            self.svc.execute_batch_offload,
            CIRCUIT, self.jobs, [backend("a"), backend("b")], {"a": self.good_result},
        )
        self.assertEqual((exc.code, exc.path), ("invalid_executors", "executors"))
        # 多键
        exc = execution_err(
            self.svc.execute_batch_offload,
            CIRCUIT, self.jobs, self.backends,
            {"a": self.good_result, "b": self.good_result},
        )
        self.assertEqual((exc.code, exc.path), ("invalid_executors", "executors"))
        # 错键
        exc = execution_err(
            self.svc.execute_batch_offload,
            CIRCUIT, self.jobs, self.backends, {"x": self.good_result},
        )
        self.assertEqual((exc.code, exc.path), ("invalid_executors", "executors"))

    def test_executor_values_must_be_callable(self):
        for bad in (None, 42, "callable", {}):
            exc = execution_err(
                self.svc.execute_batch_offload,
                CIRCUIT, self.jobs, self.backends, {"a": bad},
            )
            self.assertEqual((exc.code, exc.path), ("invalid_executors", "executors"), bad)

    def test_no_executor_called_on_request_failure(self):
        called = threading.Event()

        def spy(circuit, spec):
            called.set()
            return self.good_result(circuit, spec)

        for bad_concurrency in (0, True):
            called.clear()
            with self.assertRaises(BatchExecutionError):
                self.svc.execute_batch_offload(
                    CIRCUIT, self.jobs, self.backends, {"a": spy}, bad_concurrency,
                )
            self.assertFalse(called.is_set())
        with self.assertRaises(OffloadExecutionError):
            self.svc.execute_batch_offload(
                CIRCUIT, self.jobs, self.backends, {"a": 42},
            )
        self.assertFalse(called.is_set())

    def test_no_partial_response_on_request_failure(self):
        with self.assertRaises((CircuitValidationError, OffloadPlanningError,
                                BatchExecutionError, OffloadExecutionError)):
            self.svc.execute_batch_offload(
                {"qubit_count": -1}, self.jobs, self.backends, {"a": self.good_result},
            )


class InputImmutabilityTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_inputs_not_modified(self):
        circuit = copy.deepcopy(CIRCUIT)
        jobs = [
            job("a", values={"theta": 0.0}),
            job("b", values={"theta": 0.5}, shots=3, seed=1),
            job("bad", ["ZZ"]),
        ]
        backends = [backend("a", slots=1), backend("b", slots=2)]
        executors = {"a": local_executor(self.svc), "b": local_executor(self.svc)}
        circuit_copy = copy.deepcopy(circuit)
        jobs_copy = copy.deepcopy(jobs)
        backends_copy = copy.deepcopy(backends)
        self.svc.execute_batch_offload(
            circuit, jobs, backends, executors, max_concurrency=3,
        )
        self.assertEqual(circuit, circuit_copy)
        self.assertEqual(jobs, jobs_copy)
        self.assertEqual(backends, backends_copy)


class ExecutionErrorHierarchyTest(unittest.TestCase):
    def test_is_value_error_with_code_and_path(self):
        exc = OffloadExecutionError("invalid_executors", "executors")
        self.assertIsInstance(exc, ValueError)
        self.assertEqual(exc.code, "invalid_executors")
        self.assertEqual(exc.path, "executors")
        self.assertIn("invalid_executors", str(exc))

    def test_exported_from_package_and_service(self):
        self.assertIs(PkgOffloadExecutionError, OffloadExecutionError)


if __name__ == "__main__":
    unittest.main()
