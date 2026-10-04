"""``Service.diagnose_batch_offload`` 的契约测试。

诊断入口与 ``plan_batch_offload`` 共用规划输入与请求级校验，只审计
选择过程、不执行仿真；这些测试覆盖 plan 逐值一致、诊断项形状、逐作业
状态推进、候选原因/预算明细、rejected 项、请求级校验同构以及 JSON
原生与输入不变性。
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

BUDGET_KEYS = (
    "max_state_bytes",
    "max_circuit_evaluations",
    "max_gate_applications",
    "max_total_shots",
)
REQUIREMENT_KEYS = (
    "representation",
    "state_bytes",
    "circuit_evaluations",
    "gate_applications",
    "total_shots",
)


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


def mixed_jobs():
    return [
        job("ok", values={"theta": 0.0}),
        job("missing-binding"),
        job("bad-observable", ["ZZZ"], values={"theta": 0.0}),
        job("sampled", values={"theta": 0.0}, shots=50, seed=3),
        job("noisy", values={"theta": 0.0},
            noise={"single_qubit_depolarizing": 0.05}),
    ]


def mixed_backends():
    return [
        backend("sv", slots=1, reps=["state_vector"]),
        backend("dm", slots=8, reps=["density_matrix"],
                max_total_shots=10, max_state_bytes=1),
        backend("both", slots=1),
    ]


class PlanEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def assert_plan_equals(self, circuit, jobs, backends):
        plan_out = self.svc.plan_batch_offload(
            copy.deepcopy(circuit), copy.deepcopy(jobs), copy.deepcopy(backends)
        )
        diag_out = self.svc.diagnose_batch_offload(circuit, jobs, backends)
        self.assertEqual(set(diag_out), {"plan", "diagnostics"})
        self.assertEqual(diag_out["plan"], plan_out)
        # 序列化后字段、顺序、值仍然一致。
        self.assertEqual(json.loads(json.dumps(diag_out["plan"])), plan_out)
        return diag_out

    def test_plan_value_identical_across_mixed_outcomes(self):
        out = self.assert_plan_equals(CIRCUIT, mixed_jobs(), mixed_backends())
        statuses = [item["status"] for item in out["plan"]["results"]]
        self.assertIn("assigned", statuses)
        self.assertIn("rejected", statuses)
        self.assertIn("no_eligible_backend", statuses)

    def test_plan_identical_for_balancing_scenario(self):
        backends = [backend("a", slots=2), backend("b", slots=3)]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(6)]
        self.assert_plan_equals(CIRCUIT, jobs, backends)

    def test_plan_identical_for_single_backend_edge_cases(self):
        self.assert_plan_equals(
            CIRCUIT,
            [job("j", ["ZZ"], values={"theta": 0.0}, shots=10)],
            [backend("a", slots=1, max_state_bytes=64, max_circuit_evaluations=1,
                     max_gate_applications=3, max_total_shots=10)],
        )
        self.assert_plan_equals(
            {"qubit_count": 21},
            [job("big", ["I" * 21])],
            [backend("a")],
        )


class DiagnosticsShapeTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_diagnostics_follow_jobs_order(self):
        jobs = mixed_jobs()
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, mixed_backends())
        self.assertEqual(len(out["diagnostics"]), len(jobs))
        self.assertEqual([d["id"] for d in out["diagnostics"]], [j["id"] for j in jobs])

    def test_assigned_item_shape(self):
        out = self.svc.diagnose_batch_offload(
            CIRCUIT, [job("j", values={"theta": 0.0})], [backend("a")]
        )
        diag = out["diagnostics"][0]
        self.assertEqual(
            set(diag),
            {"id", "status", "requirements", "selected_backend_id", "candidates"},
        )
        self.assertEqual(diag["status"], "assigned")
        self.assertEqual(diag["selected_backend_id"], "a")
        plan_item = out["plan"]["results"][0]
        self.assertEqual(diag["requirements"], plan_item["requirements"])
        self.assertEqual(set(diag["requirements"]), set(REQUIREMENT_KEYS))

    def test_rejected_item_shape(self):
        out = self.svc.diagnose_batch_offload(
            CIRCUIT, mixed_jobs(), mixed_backends()
        )
        diag = next(d for d in out["diagnostics"] if d["status"] == "rejected")
        self.assertEqual(
            set(diag),
            {"id", "status", "requirements", "selected_backend_id",
             "candidates", "validation_error"},
        )
        self.assertIsNone(diag["requirements"])
        self.assertIsNone(diag["selected_backend_id"])
        self.assertEqual(diag["candidates"], [])
        plan_item = next(
            r for r in out["plan"]["results"] if r["id"] == diag["id"]
        )
        self.assertEqual(diag["validation_error"], plan_item["validation_error"])
        self.assertEqual(
            set(diag["validation_error"]),
            {"type", "code", "path", "message"},
        )

    def test_no_eligible_item_keeps_requirements_and_candidates(self):
        tight = backend("a", slots=2, max_total_shots=99)
        jobs = [
            job("exact", ["ZZ"], values={"theta": 0.0}),
            job("sampled", ["ZZ", "XX"], values={"theta": 0.0}, shots=50),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, [tight])
        diag = out["diagnostics"][1]
        self.assertEqual(diag["status"], "no_eligible_backend")
        self.assertIsNotNone(diag["requirements"])
        self.assertEqual(diag["requirements"]["total_shots"], 100)
        self.assertIsNone(diag["selected_backend_id"])
        self.assertEqual(len(diag["candidates"]), 1)
        plan_item = out["plan"]["results"][1]
        self.assertIsNone(plan_item["requirements"])
        self.assertEqual(
            [c["reasons"] for c in diag["candidates"]], plan_item["reasons"]
        )


class CandidatesTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_candidates_follow_backends_order_and_field_shape(self):
        backends = mixed_backends()
        out = self.svc.diagnose_batch_offload(
            CIRCUIT, [job("j", values={"theta": 0.0})], backends
        )
        candidates = out["diagnostics"][0]["candidates"]
        self.assertEqual([c["backend_id"] for c in candidates],
                         [b["id"] for b in backends])
        for cand, backend_spec in zip(candidates, backends):
            self.assertEqual(
                set(cand),
                {"backend_id", "assigned_before", "slots", "eligible",
                 "reasons", "budgets"},
            )
            self.assertEqual(cand["slots"], backend_spec["slots"])
            self.assertIsInstance(cand["assigned_before"], int)
            self.assertIsInstance(cand["eligible"], bool)
            self.assertEqual(cand["eligible"], not cand["reasons"])
            self.assertEqual(list(cand["budgets"]), list(BUDGET_KEYS))
            for budget in cand["budgets"].values():
                self.assertEqual(set(budget), {"required", "limit", "exceeded"})
                self.assertIsInstance(budget["required"], int)
                self.assertIsInstance(budget["limit"], int)
                self.assertIsInstance(budget["exceeded"], bool)

    def test_first_job_sees_zero_assigned_before_everywhere(self):
        backends = [backend("a", slots=2), backend("b", slots=3)]
        out = self.svc.diagnose_batch_offload(
            CIRCUIT, [job("j0", values={"theta": 0.0})], backends
        )
        candidates = out["diagnostics"][0]["candidates"]
        self.assertEqual([c["assigned_before"] for c in candidates], [0, 0])

    def test_assigned_before_tracks_per_job_progression(self):
        # slots 2 vs 3 的均衡序列：a,b,b,a,(none)
        backends = [backend("a", slots=2), backend("b", slots=3)]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(6)]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        diags = out["diagnostics"]
        before = [[c["assigned_before"] for c in d["candidates"]] for d in diags]
        self.assertEqual(
            before,
            [[0, 0], [1, 0], [1, 1], [1, 2], [2, 2], [2, 3]],
        )
        self.assertEqual(
            [d["selected_backend_id"] for d in diags[:5]],
            ["a", "b", "b", "a", "b"],
        )
        self.assertIsNone(diags[5]["selected_backend_id"])
        # 满载时两个后端都报 no_slot。
        self.assertEqual(
            [c["reasons"] for c in diags[5]["candidates"]],
            [["no_slot"], ["no_slot"]],
        )

    def test_successful_assignment_keeps_all_backends_with_states(self):
        backends = [
            backend("sv", slots=1, reps=["state_vector"]),
            backend("dm", slots=1, reps=["density_matrix"]),
        ]
        jobs = [
            job("sv-job", values={"theta": 0.0}),
            job("dm-job", values={"theta": 0.0},
                noise={"single_qubit_depolarizing": 0.1}),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        d0, d1 = out["diagnostics"]
        self.assertEqual(d0["selected_backend_id"], "sv")
        self.assertEqual(len(d0["candidates"]), 2)
        self.assertTrue(d0["candidates"][0]["eligible"])
        self.assertFalse(d0["candidates"][1]["eligible"])
        # 第二个作业看到 sv 已被占用（assigned_before=1，slot=1 → no_slot）。
        sv_cand, dm_cand = d1["candidates"]
        self.assertEqual(sv_cand["assigned_before"], 1)
        self.assertEqual(
            sv_cand["reasons"],
            ["unsupported_representation", "no_slot"],
        )
        self.assertFalse(sv_cand["eligible"])
        self.assertEqual(dm_cand["reasons"], [])
        self.assertTrue(dm_cand["eligible"])
        self.assertEqual(d1["selected_backend_id"], "dm")

    def test_reasons_order_unsupported_slot_then_budget_keys(self):
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
        candidates = out["diagnostics"][1]["candidates"]
        self.assertEqual(
            [c["reasons"] for c in candidates],
            [
                [
                    "unsupported_representation",
                    "max_state_bytes",
                    "max_circuit_evaluations",
                    "max_gate_applications",
                ],
                ["no_slot"],
            ],
        )
        self.assertFalse(any(c["eligible"] for c in candidates))

    def test_budgets_required_limit_exceeded(self):
        # 精确作业：state_bytes=64, evaluations=1, gates=3, total_shots=0。
        b = backend("a", slots=1, max_state_bytes=64, max_circuit_evaluations=1,
                    max_gate_applications=3, max_total_shots=10)
        j = job("j", ["ZZ"], values={"theta": 0.0}, shots=10)
        out = self.svc.diagnose_batch_offload(CIRCUIT, [j], [b])
        budgets = out["diagnostics"][0]["candidates"][0]["budgets"]
        self.assertEqual(
            budgets,
            {
                "max_state_bytes": {"required": 64, "limit": 64, "exceeded": False},
                "max_circuit_evaluations": {"required": 1, "limit": 1, "exceeded": False},
                "max_gate_applications": {"required": 3, "limit": 3, "exceeded": False},
                "max_total_shots": {"required": 10, "limit": 10, "exceeded": False},
            },
        )
        self.assertTrue(out["diagnostics"][0]["candidates"][0]["eligible"])

    def test_budget_exceeded_flags_match_reasons(self):
        tight = backend("a", slots=2, max_total_shots=99)
        j = job("sampled", ["ZZ", "XX"], values={"theta": 0.0}, shots=50)
        out = self.svc.diagnose_batch_offload(CIRCUIT, [j], [tight])
        cand = out["diagnostics"][0]["candidates"][0]
        self.assertFalse(cand["eligible"])
        self.assertEqual(cand["reasons"], ["max_total_shots"])
        self.assertFalse(cand["budgets"]["max_state_bytes"]["exceeded"])
        self.assertFalse(cand["budgets"]["max_circuit_evaluations"]["exceeded"])
        self.assertFalse(cand["budgets"]["max_gate_applications"]["exceeded"])
        total = cand["budgets"]["max_total_shots"]
        self.assertEqual(total, {"required": 100, "limit": 99, "exceeded": True})

    def test_selected_backend_is_min_ratio_eligible_candidate(self):
        backends = [backend("a", slots=2), backend("b", slots=3), backend("c", slots=5)]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(4)]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        for diag in out["diagnostics"]:
            eligible = [c for c in diag["candidates"] if c["eligible"]]
            self.assertTrue(eligible)
            # 手工复核最小 assigned_before/slots（交叉乘法）：选中者的比例
            # 严格小于或相等（相等时输入顺序在前）每个其他合格候选。
            selected = next(
                c for c in diag["candidates"]
                if c["backend_id"] == diag["selected_backend_id"]
            )
            selected_index = diag["candidates"].index(selected)
            for index, cand in enumerate(diag["candidates"]):
                if not cand["eligible"] or cand is selected:
                    continue
                self.assertLessEqual(
                    selected["assigned_before"] * cand["slots"],
                    cand["assigned_before"] * selected["slots"],
                )
                if (
                    selected["assigned_before"] * cand["slots"]
                    == cand["assigned_before"] * selected["slots"]
                ):
                    self.assertLess(selected_index, index)
            self.assertIn(selected, eligible)


class RequestValidationParityTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.good_jobs = [job("j", values={"theta": 0.0})]

    def assert_same_failure(self, circuit, jobs, backends):
        """两个入口在同一请求级失败上抛同类型同 code/path 的异常。"""
        with self.assertRaises(Exception) as ctx_plan:
            self.svc.plan_batch_offload(
                copy.deepcopy(circuit), copy.deepcopy(jobs), copy.deepcopy(backends)
            )
        with self.assertRaises(Exception) as ctx_diag:
            self.svc.diagnose_batch_offload(circuit, jobs, backends)
        self.assertEqual(type(ctx_diag.exception), type(ctx_plan.exception))
        self.assertEqual(ctx_diag.exception.code, ctx_plan.exception.code)
        self.assertEqual(ctx_diag.exception.path, ctx_plan.exception.path)
        self.assertEqual(str(ctx_diag.exception), str(ctx_plan.exception))

    def test_circuit_error_priority(self):
        self.assert_same_failure({"qubit_count": -1}, None, None)
        self.assert_same_failure("nope", [], [])

    def test_invalid_backends(self):
        for bad in (None, [], {}, "backends", 1, True):
            self.assert_same_failure(CIRCUIT, self.good_jobs, bad)

    def test_invalid_backend_and_duplicate_id(self):
        b = backend("a")
        b["bogus"] = 1
        self.assert_same_failure(CIRCUIT, self.good_jobs, [b])
        self.assert_same_failure(
            CIRCUIT, self.good_jobs, [backend("x"), backend("y"), backend("x")]
        )

    def test_jobs_structure_errors(self):
        for bad in (None, [], {}, 1):
            self.assert_same_failure(CIRCUIT, bad, [backend("a")])
        self.assert_same_failure(
            CIRCUIT, [{"observables": ["ZZ"]}], [backend("a")]
        )
        self.assert_same_failure(
            CIRCUIT,
            [job("x", values={"theta": 0.0}), job("x", values={"theta": 0.0})],
            [backend("a")],
        )

    def test_validation_order_circuit_backends_jobs(self):
        self.assert_same_failure(CIRCUIT, None, None)
        self.assert_same_failure(CIRCUIT, [], [])
        self.assert_same_failure(CIRCUIT, "not-jobs", backend("a"))
        self.assert_same_failure(CIRCUIT, "not-jobs", [backend("a")])

    def test_no_partial_result_on_request_failure(self):
        for bad in (
            ({"qubit_count": -1}, self.good_jobs, [backend("a")]),
            (CIRCUIT, None, None),
            (CIRCUIT, "not-jobs", [backend("a")]),
        ):
            with self.assertRaises(
                (CircuitValidationError, OffloadPlanningError, BatchExecutionError)
            ):
                self.svc.diagnose_batch_offload(*bad)

    def test_semantic_failure_does_not_block_later_jobs(self):
        jobs = [
            job("bad"),
            job("ok", values={"theta": 0.0}),
        ]
        out = self.svc.diagnose_batch_offload(CIRCUIT, jobs, [backend("a", slots=1)])
        self.assertEqual([d["status"] for d in out["diagnostics"]],
                         ["rejected", "assigned"])
        self.assertEqual(out["plan"]["results"][1]["backend_id"], "a")


class OutputContractTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_output_is_json_native(self):
        out = self.svc.diagnose_batch_offload(
            CIRCUIT, mixed_jobs(), mixed_backends()
        )
        json.dumps(out)

    def test_inputs_not_modified(self):
        jobs = mixed_jobs()
        backends = mixed_backends()
        circuit_copy = copy.deepcopy(CIRCUIT)
        jobs_copy = copy.deepcopy(jobs)
        backends_copy = copy.deepcopy(backends)
        self.svc.diagnose_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(CIRCUIT, circuit_copy)
        self.assertEqual(jobs, jobs_copy)
        self.assertEqual(backends, backends_copy)

    def test_diagnose_does_not_execute_simulation(self):
        out = self.svc.diagnose_batch_offload(
            CIRCUIT,
            [job("a", values={"theta": 0.0}, shots=10, seed=1)],
            [backend("x")],
        )
        diag = out["diagnostics"][0]
        self.assertNotIn("result", diag)
        self.assertNotIn("expectation", diag["requirements"])
        self.assertNotIn("counts", diag["requirements"])

    def test_repeated_calls_are_identical(self):
        jobs = mixed_jobs()
        backends = mixed_backends()
        first = self.svc.diagnose_batch_offload(
            copy.deepcopy(CIRCUIT), copy.deepcopy(jobs), copy.deepcopy(backends)
        )
        second = self.svc.diagnose_batch_offload(
            copy.deepcopy(CIRCUIT), copy.deepcopy(jobs), copy.deepcopy(backends)
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
