"""Service.diagnose_batch_offload 的契约测试。

诊断入口必须：
- plan 与同输入 plan_batch_offload 的结果逐值一致；
- diagnostics 按 jobs 顺序与 plan results 一一对应，记录逐作业推进、
  占用 slot 之前的全部后端候选快照；
- 与 plan_batch_offload 共享请求级校验顺序、异常类型、code、path；
- 只输出 JSON 原生类型、不修改输入、结果确定。
"""

import copy
import json
import unittest

from qubitfabric.service import (
    BatchExecutionError,
    CircuitValidationError,
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

_BUDGET_KEYS = (
    "max_state_bytes",
    "max_circuit_evaluations",
    "max_gate_applications",
    "max_total_shots",
)
_REQUIREMENT_FIELDS = {
    "representation",
    "state_bytes",
    "circuit_evaluations",
    "gate_applications",
    "total_shots",
}
_CANDIDATE_FIELDS = {
    "backend_id",
    "assigned_before",
    "slots",
    "eligible",
    "reasons",
    "budgets",
}
_VALID_DIAG_FIELDS = {
    "id",
    "status",
    "requirements",
    "selected_backend_id",
    "candidates",
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


class PlanParityTest(unittest.TestCase):
    """plan 必须与 plan_batch_offload 完全一致（字段、顺序、值）。"""

    def setUp(self):
        self.svc = Service()

    def assert_plan_parity(self, circuit, jobs, backends):
        plan_only = self.svc.plan_batch_offload(circuit, jobs, backends)
        diag = self.svc.diagnose_batch_offload(circuit, jobs, backends)
        self.assertEqual(set(diag), {"plan", "diagnostics"})
        self.assertEqual(diag["plan"], plan_only)
        # 再做一次 JSON 往返，确认逐值可序列化且结构稳定。
        self.assertEqual(json.loads(json.dumps(diag))["plan"], plan_only)
        return diag

    def test_mixed_statuses_plan_matches_verbatim(self):
        jobs = [
            job("ok", values={"theta": 0.0}),
            job("missing-binding"),
            job("sampled", values={"theta": 0.0}, shots=50, seed=3),
            job("seed-no-shots", values={"theta": 0.0}, seed=3),
            job("noisy", values={"theta": 0.0},
                noise={"single_qubit_depolarizing": 0.1}),
        ]
        backends = [
            backend("sv", slots=1, reps=["state_vector"]),
            backend("dm", slots=8, reps=["density_matrix"], max_total_shots=1000),
        ]
        diag = self.assert_plan_parity(CIRCUIT, jobs, backends)
        self.assertEqual(len(diag["diagnostics"]), len(jobs))

    def test_ratio_balancing_plan_matches_verbatim(self):
        backends = [backend("a", slots=2), backend("b", slots=3)]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(6)]
        self.assert_plan_parity(CIRCUIT, jobs, backends)

    def test_all_backends_full_plan_matches(self):
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(3)]
        self.assert_plan_parity(CIRCUIT, jobs, [backend("a", slots=1), backend("b", slots=1)])

    def test_all_rejected_plan_matches(self):
        jobs = [job("bad"), job("bad2", ["ZZZ"], values={"theta": 0.0})]
        self.assert_plan_parity(CIRCUIT, jobs, [backend("a")])

    def test_deterministic_across_calls(self):
        jobs = [
            job("a", values={"theta": 0.0}),
            job("b", values={"theta": 0.0}, shots=4, seed=1),
        ]
        backends = [backend("a", slots=2), backend("b", slots=3)]
        first = self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        second = self.svc.diagnose_batch_offload(
            copy.deepcopy(CIRCUIT), copy.deepcopy(jobs), copy.deepcopy(backends),
        )
        self.assertEqual(first, second)


class DiagnosticsAlignmentTest(unittest.TestCase):
    """diagnostics 与 plan results 同序同 id/status，且形态符合契约。"""

    def setUp(self):
        self.svc = Service()
        self.jobs = [
            job("ok", values={"theta": 0.0}),
            job("bad"),
            job("sampled", values={"theta": 0.0}, shots=50, seed=3),
        ]
        self.backends = [backend("a", slots=4), backend("b", slots=1)]
        self.diag = self.svc.diagnose_batch_offload(
            CIRCUIT, self.jobs, self.backends,
        )

    def test_diagnostics_align_one_to_one_with_plan_results(self):
        results = self.diag["plan"]["results"]
        diagnostics = self.diag["diagnostics"]
        self.assertEqual(len(diagnostics), len(results))
        for result, diag_item in zip(results, diagnostics):
            self.assertEqual(diag_item["id"], result["id"])
            self.assertEqual(diag_item["status"], result["status"])

    def test_assigned_item_shape(self):
        result = self.diag["plan"]["results"][0]
        item = self.diag["diagnostics"][0]
        self.assertEqual(result["status"], "assigned")
        self.assertEqual(set(item), _VALID_DIAG_FIELDS)
        self.assertEqual(item["selected_backend_id"], result["backend_id"])
        self.assertEqual(item["requirements"], result["requirements"])
        self.assertEqual(set(item["requirements"]), _REQUIREMENT_FIELDS)

    def test_rejected_item_shape(self):
        result = self.diag["plan"]["results"][1]
        item = self.diag["diagnostics"][1]
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(
            set(item), _VALID_DIAG_FIELDS | {"validation_error"},
        )
        self.assertIsNone(item["requirements"])
        self.assertIsNone(item["selected_backend_id"])
        self.assertEqual(item["candidates"], [])
        self.assertEqual(item["validation_error"], result["validation_error"])
        self.assertEqual(
            set(item["validation_error"]),
            {"type", "code", "path", "message"},
        )

    def test_no_eligible_item_shape_keeps_requirements(self):
        # b 只有 1 slot，被第一个作业占用；sampled 需要 1000 total_shots，
        # a 的 1e9 足够但... 改为显式构造无候选场景。
        tight = [
            backend("a", slots=1, max_total_shots=99),
        ]
        jobs = [
            job("fill", ["ZZ"], values={"theta": 0.0}),
            job("homeless", ["ZZ", "XX"], values={"theta": 0.0}, shots=50),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, tight)
        result = out["plan"]["results"][1]
        item = out["diagnostics"][1]
        self.assertEqual(result["status"], "no_eligible_backend")
        self.assertIsNone(result["requirements"])  # 规划结果仍为 None
        self.assertEqual(set(item), _VALID_DIAG_FIELDS)
        self.assertIsNone(item["selected_backend_id"])
        self.assertIsNotNone(item["requirements"])
        self.assertEqual(set(item["requirements"]), _REQUIREMENT_FIELDS)
        self.assertEqual(item["requirements"]["total_shots"], 100)
        self.assertEqual(len(item["candidates"]), 1)


class CandidateSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_candidates_follow_backend_order_and_record_pre_assignment_state(self):
        backends = [backend("a", slots=2), backend("b", slots=3)]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(6)]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        items = out["diagnostics"]

        expected_before = [
            [0, 0],  # j0 -> a
            [1, 0],  # j1 -> b
            [1, 1],  # j2 -> b (1/3 < 1/2)
            [1, 2],  # j3 -> a (1/2 < 2/3)
            [2, 2],  # j4 -> b (2/3 < 1)
            [2, 3],  # j5 -> 两个后端均满
        ]
        chosen = ["a", "b", "b", "a", "b", None]
        for item, before, want_backend in zip(items, expected_before, chosen):
            self.assertEqual([c["backend_id"] for c in item["candidates"]], ["a", "b"])
            self.assertEqual([c["assigned_before"] for c in item["candidates"]], before)
            self.assertEqual([c["slots"] for c in item["candidates"]], [2, 3])
            self.assertEqual(item["selected_backend_id"], want_backend)

        # 最后一项两个后端都不再 eligible，原因为 no_slot。
        last = items[5]
        self.assertEqual(last["status"], "no_eligible_backend")
        self.assertTrue(all(not c["eligible"] for c in last["candidates"]))
        self.assertEqual([c["reasons"] for c in last["candidates"]],
                         [["no_slot"], ["no_slot"]])

    def test_selection_replayable_from_assigned_before_and_slots(self):
        """调用方仅凭 candidates 的 assigned_before/slots 即可复核选择。"""
        backends = [backend("a", slots=2), backend("b", slots=3)]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(6)]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)

        counts = {b["id"]: 0 for b in backends}
        for item in out["diagnostics"]:
            if item["status"] == "rejected":
                self.assertEqual(item["candidates"], [])
                continue
            candidates = item["candidates"]
            # eligible 当且仅当 reasons 为空。
            for cand in candidates:
                self.assertEqual(cand["eligible"], cand["reasons"] == [])
                self.assertEqual(cand["assigned_before"], counts[cand["backend_id"]])
            eligible = [c for c in candidates if c["eligible"]]
            # 交叉乘法选 assigned_before/slots 最小者，相等取输入靠前者。
            replay_choice = None
            best_num = 0
            best_den = 1
            for cand in eligible:
                num, den = cand["assigned_before"], cand["slots"]
                if replay_choice is None or num * best_den < best_num * den:
                    replay_choice = cand["backend_id"]
                    best_num, best_den = num, den
            self.assertEqual(replay_choice, item["selected_backend_id"])
            if replay_choice is not None:
                counts[replay_choice] += 1

        self.assertEqual(counts, {"a": 2, "b": 3})

    def test_successful_item_keeps_all_backends_even_ineligible_ones(self):
        noisy = job("n", values={"theta": 0.0},
                    noise={"single_qubit_depolarizing": 0.1})
        backends = [
            backend("sv", reps=["state_vector"]),
            backend("dm", reps=["density_matrix"]),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, [noisy], backends)
        item = out["diagnostics"][0]
        self.assertEqual(item["status"], "assigned")
        self.assertEqual(item["selected_backend_id"], "dm")
        self.assertEqual(len(item["candidates"]), 2)
        sv, dm = item["candidates"]
        self.assertEqual(sv["backend_id"], "sv")
        self.assertFalse(sv["eligible"])
        self.assertEqual(sv["reasons"], ["unsupported_representation"])
        self.assertTrue(dm["eligible"])
        self.assertEqual(dm["reasons"], [])

    def test_candidate_shape_and_budgets(self):
        b = backend("a", slots=2, max_total_shots=99)
        jobs = [
            job("exact", ["ZZ"], values={"theta": 0.0}),
            job("sampled", ["ZZ", "XX"], values={"theta": 0.0}, shots=50),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, [b])
        exact, sampled = out["diagnostics"]

        cand = exact["candidates"][0]
        self.assertEqual(set(cand), _CANDIDATE_FIELDS)
        self.assertEqual(cand["assigned_before"], 0)
        self.assertEqual(cand["slots"], 2)
        self.assertTrue(cand["eligible"])
        self.assertEqual(list(cand["budgets"]), list(_BUDGET_KEYS))
        for key, entry in cand["budgets"].items():
            self.assertEqual(set(entry), {"required", "limit", "exceeded"})
            self.assertEqual(entry["limit"], b[key])
            self.assertIsInstance(entry["exceeded"], bool)
        # 2 量子位状态向量：4 元素 * 16 字节；3 个操作；精确模式 shots 为 0。
        self.assertEqual(cand["budgets"]["max_state_bytes"]["required"], 64)
        self.assertEqual(cand["budgets"]["max_circuit_evaluations"]["required"], 1)
        self.assertEqual(cand["budgets"]["max_gate_applications"]["required"], 3)
        self.assertEqual(cand["budgets"]["max_total_shots"]["required"], 0)
        self.assertFalse(cand["budgets"]["max_total_shots"]["exceeded"])

        cand2 = sampled["candidates"][0]
        self.assertEqual(cand2["assigned_before"], 1)
        self.assertEqual(sampled["requirements"]["total_shots"], 100)
        shots_entry = cand2["budgets"]["max_total_shots"]
        self.assertEqual(shots_entry["required"], 100)
        self.assertEqual(shots_entry["limit"], 99)
        self.assertTrue(shots_entry["exceeded"])
        self.assertFalse(cand2["eligible"])
        self.assertEqual(cand2["reasons"], ["max_total_shots"])
        self.assertIsNone(sampled["selected_backend_id"])

    def test_requirement_equal_to_limit_is_not_exceeded_and_eligible(self):
        b = backend("a", slots=1, max_state_bytes=64, max_circuit_evaluations=1,
                    max_gate_applications=3, max_total_shots=10)
        j = job("j", ["ZZ"], values={"theta": 0.0}, shots=10)
        out = self.svc.diagnose_batch_offload(CIRCUIT, [j], [b])
        item = out["diagnostics"][0]
        self.assertEqual(item["status"], "assigned")
        cand = item["candidates"][0]
        self.assertTrue(cand["eligible"])
        self.assertEqual(cand["reasons"], [])
        for entry in cand["budgets"].values():
            self.assertFalse(entry["exceeded"])

    def test_reasons_order_representation_slot_then_budget_keys(self):
        a = backend(
            "a", slots=1, reps=["density_matrix"],
            max_state_bytes=1, max_circuit_evaluations=0,
            max_gate_applications=0, max_total_shots=0,
        )
        b = backend("b", slots=1)
        jobs = [
            job("fill", values={"theta": 0.0}),
            job("homeless", values={"theta": 0.0}),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, [a, b])
        item = out["diagnostics"][1]
        self.assertEqual(item["status"], "no_eligible_backend")
        self.assertEqual([c["reasons"] for c in item["candidates"]], [
            [
                "unsupported_representation",
                "max_state_bytes",
                "max_circuit_evaluations",
                "max_gate_applications",
            ],
            ["no_slot"],
        ])
        # 候选原因与规划结果 reasons 逐值一致。
        plan_item = out["plan"]["results"][1]
        self.assertEqual(
            [c["reasons"] for c in item["candidates"]],
            plan_item["reasons"],
        )
        # 预算快照与原因中的超限键一致。
        for cand in item["candidates"]:
            exceeded_keys = [
                key for key in _BUDGET_KEYS if cand["budgets"][key]["exceeded"]
            ]
            budget_reasons = [r for r in cand["reasons"] if r in _BUDGET_KEYS]
            self.assertEqual(exceeded_keys, budget_reasons)

    def test_rejected_job_does_not_consume_slot_in_later_snapshot(self):
        jobs = [
            job("bad"),
            job("ok", values={"theta": 0.0}),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, [backend("a", slots=1)])
        rejected, assigned_item = out["diagnostics"]
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(assigned_item["candidates"][0]["assigned_before"], 0)
        self.assertTrue(assigned_item["candidates"][0]["eligible"])

    def test_no_eligible_requirements_match_resource_estimator(self):
        j = job("homeless", ["ZZ", "XX"], values={"theta": 0.0}, shots=50, seed=7)
        out = self.svc.diagnose_batch_offload(
            CIRCUIT, [j], [backend("a", slots=1, max_total_shots=1)],
        )
        item = out["diagnostics"][0]
        self.assertEqual(item["status"], "no_eligible_backend")
        resources = self.svc.estimate_resources(
            CIRCUIT,
            {
                "type": "sampled_expectation",
                "observables": j["observables"],
                "values": j["values"],
                "shots": j["shots"],
                "seed": j["seed"],
            },
        )
        req = item["requirements"]
        for key in _REQUIREMENT_FIELDS:
            self.assertEqual(req[key], resources[key])


class RequestValidationParityTest(unittest.TestCase):
    """请求级校验顺序、异常类型、code、path 与 plan_batch_offload 相同。"""

    def setUp(self):
        self.svc = Service()
        self.good_jobs = [job("j", values={"theta": 0.0})]

    def assert_same_failure(self, circuit, jobs, backends, expected):
        with self.assertRaises(Exception) as plan_ctx:
            self.svc.plan_batch_offload(circuit, jobs, backends)
        with self.assertRaises(Exception) as diag_ctx:
            self.svc.diagnose_batch_offload(circuit, jobs, backends)
        self.assertIs(type(diag_ctx.exception), type(plan_ctx.exception))
        self.assertEqual(diag_ctx.exception.code, plan_ctx.exception.code)
        self.assertEqual(diag_ctx.exception.path, plan_ctx.exception.path)
        if expected is not None:
            self.assertEqual(
                (type(diag_ctx.exception).__name__,
                 diag_ctx.exception.code, diag_ctx.exception.path),
                expected,
            )

    def test_circuit_error_has_priority(self):
        self.assert_same_failure(
            {"qubit_count": -1}, None, None,
            ("CircuitValidationError", "invalid_value", "qubit_count"),
        )
        self.assert_same_failure("nope", [], [], None)

    def test_backend_errors(self):
        self.assert_same_failure(
            CIRCUIT, self.good_jobs, None,
            ("OffloadPlanningError", "invalid_backends", "backends"),
        )
        self.assert_same_failure(
            CIRCUIT, self.good_jobs, [],
            ("OffloadPlanningError", "invalid_backends", "backends"),
        )
        self.assert_same_failure(
            CIRCUIT, self.good_jobs, ["x"],
            ("OffloadPlanningError", "invalid_backend", "backends[0]"),
        )
        bad = backend("a")
        bad["bogus"] = 1
        self.assert_same_failure(
            CIRCUIT, self.good_jobs, [bad],
            ("OffloadPlanningError", "invalid_backend", "backends[0].bogus"),
        )
        self.assert_same_failure(
            CIRCUIT, self.good_jobs,
            [backend("x"), backend("y"), backend("x")],
            ("OffloadPlanningError", "duplicate_backend_id", "backends[2].id"),
        )

    def test_jobs_structure_errors(self):
        self.assert_same_failure(
            CIRCUIT, None, [backend("a")],
            ("BatchExecutionError", "invalid_batch", "jobs"),
        )
        self.assert_same_failure(
            CIRCUIT, [{"observables": ["ZZ"]}], [backend("a")],
            ("BatchExecutionError", "invalid_job", "jobs[0].id"),
        )
        dup = [job("x", values={"theta": 0.0}), job("x", values={"theta": 0.0})]
        self.assert_same_failure(
            CIRCUIT, dup, [backend("a")],
            ("BatchExecutionError", "duplicate_job_id", "jobs[1].id"),
        )

    def test_validation_order_circuit_backends_jobs(self):
        # backends 错误优先于 jobs 结构错误。
        self.assert_same_failure(CIRCUIT, None, None,
                                 ("OffloadPlanningError", "invalid_backends", "backends"))
        self.assert_same_failure(CIRCUIT, [], [],
                                 ("OffloadPlanningError", "invalid_backends", "backends"))
        self.assert_same_failure(CIRCUIT, "not-jobs", backend("a"),
                                 ("OffloadPlanningError", "invalid_backends", "backends"))
        self.assert_same_failure(CIRCUIT, "not-jobs", [backend("a")],
                                 ("BatchExecutionError", "invalid_batch", "jobs"))

    def test_no_partial_result_on_request_failure(self):
        for args in (
            ({"qubit_count": -1}, self.good_jobs, [backend("a")]),
            (CIRCUIT, self.good_jobs, None),
            (CIRCUIT, None, [backend("a")]),
        ):
            with self.assertRaises(
                (CircuitValidationError, OffloadPlanningError, BatchExecutionError)
            ):
                self.svc.diagnose_batch_offload(*args)


class OutputContractTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_output_is_json_native(self):
        jobs = [
            job("a", values={"theta": 0.0}),
            job("b", values={"theta": 0.0}, shots=3, seed=1),
            job("bad"),
            job("noisy", values={"theta": 0.0},
                noise={"single_qubit_depolarizing": 0.05}),
        ]
        backends = [
            backend("sv", slots=1, reps=["state_vector"]),
            backend("dm", slots=8, reps=["density_matrix"], max_total_shots=2),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        # 全部键值均可 JSON 序列化，且往返后逐值相等。
        self.assertEqual(json.loads(json.dumps(out)), out)

    def test_inputs_not_modified(self):
        jobs = [
            job("a", values={"theta": 0.0}),
            job("b", values={"theta": 0.0}, shots=3, seed=1),
            job("bad"),
        ]
        backends = [backend("a", slots=1), backend("b", slots=2)]
        circuit_copy = copy.deepcopy(CIRCUIT)
        jobs_copy = copy.deepcopy(jobs)
        backends_copy = copy.deepcopy(backends)
        self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(CIRCUIT, circuit_copy)
        self.assertEqual(jobs, jobs_copy)
        self.assertEqual(backends, backends_copy)

    def test_diagnose_does_not_mutate_plan_and_no_simulation_fields(self):
        out = self.svc.diagnose_batch_offload(
            CIRCUIT,
            [job("a", values={"theta": 0.0}, shots=10, seed=1)],
            [backend("x")],
        )
        # 不暴露期望值/计数字段。
        for item in out["diagnostics"]:
            self.assertNotIn("result", item)
            if item["requirements"] is not None:
                self.assertNotIn("expectation", item["requirements"])
                self.assertNotIn("counts", item["requirements"])


if __name__ == "__main__":
    unittest.main()
