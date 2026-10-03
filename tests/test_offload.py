import copy
import json
import unittest

from qubitfabric import OffloadPlanningError as RootOffloadPlanningError
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


def backend(
    backend_id,
    representations=("state_vector", "density_matrix"),
    slots=4,
    max_state_bytes=10**9,
    max_circuit_evaluations=10**6,
    max_gate_applications=10**6,
    max_total_shots=10**6,
):
    return {
        "id": backend_id,
        "representations": list(representations),
        "slots": slots,
        "max_state_bytes": max_state_bytes,
        "max_circuit_evaluations": max_circuit_evaluations,
        "max_gate_applications": max_gate_applications,
        "max_total_shots": max_total_shots,
    }


def job(job_id, observables=None, **extra):
    data = {"id": job_id, "observables": ["ZZ", "XX"] if observables is None else observables}
    data.update(extra)
    return data


class AssignmentSuccessTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("b0"), backend("b1")]

    def test_shape_and_order(self):
        jobs = [job("a", values={"theta": 0.0}), job("b", values={"theta": 1.0})]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, self.backends)
        self.assertEqual(set(out), {"results", "backends", "summary"})
        self.assertEqual([item["id"] for item in out["results"]], ["a", "b"])
        item = out["results"][0]
        self.assertEqual(set(item), {
            "id", "status", "assigned", "backend_id",
            "requirements", "reasons", "validation_error",
        })
        self.assertEqual(item["status"], "assigned")
        self.assertIs(item["assigned"], True)
        self.assertIn(item["backend_id"], {"b0", "b1"})
        self.assertIsNone(item["reasons"])
        self.assertIsNone(item["validation_error"])
        self.assertEqual(
            out["backends"],
            [
                {"id": "b0", "slots": 4, "assigned_count": 1},
                {"id": "b1", "slots": 4, "assigned_count": 1},
            ],
        )
        self.assertEqual(out["summary"], {"total": 2, "assigned": 2, "rejected": 0})

    def test_exact_requirements_match_resource_estimator(self):
        jobs = [job("a", values={"theta": 0.0})]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, self.backends)
        res = self.svc.estimate_resources(
            CIRCUIT, {"type": "exact_expectation",
                      "observables": ["ZZ", "XX"], "values": {"theta": 0.0}},
        )
        req = out["results"][0]["requirements"]
        self.assertEqual(req, {
            "qubit_count": res["qubit_count"],
            "representation": res["representation"],
            "state_elements": res["state_elements"],
            "state_bytes": res["state_bytes"],
            "circuit_evaluations": res["circuit_evaluations"],
            "gate_applications": res["gate_applications"],
            "total_shots": res["total_shots"],
        })

    def test_sampled_requirements_total_shots(self):
        jobs = [job("a", ["ZI", "IZ", "ZZ"], values={"theta": 0.0}, shots=100, seed=5)]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, self.backends)
        req = out["results"][0]["requirements"]
        self.assertEqual(req["representation"], "state_vector")
        self.assertEqual(req["total_shots"], 300)
        self.assertEqual(req["state_bytes"], 64)

    def test_noisy_requirements_density_matrix(self):
        jobs = [job("a", values={"theta": 0.0},
                    noise={"single_qubit_depolarizing": 0.1})]
        dms = [backend("dm", representations=("density_matrix",))]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, dms)
        item = out["results"][0]
        self.assertEqual(item["status"], "assigned")
        self.assertEqual(item["backend_id"], "dm")
        self.assertEqual(item["requirements"]["representation"], "density_matrix")
        self.assertEqual(item["requirements"]["state_elements"], 16)
        self.assertEqual(item["requirements"]["state_bytes"], 256)

    def test_output_json_native_and_input_unchanged(self):
        circuit = copy.deepcopy(CIRCUIT)
        jobs = [job("a", values={"theta": 0.0}, shots=10, seed=1)]
        backends = copy.deepcopy(self.backends)
        snapshots = copy.deepcopy((circuit, jobs, backends))
        out = self.svc.plan_batch_offload(circuit, jobs, backends)
        json.dumps(out)  # 不抛即全部为 JSON 原生类型
        self.assertEqual((circuit, jobs, backends), snapshots)


class LeastRatioSelectionTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_ties_keep_input_order(self):
        backends = [backend("b0", slots=2), backend("b1", slots=2)]
        jobs = [job("a", values={"theta": 0.0}), job("b", values={"theta": 0.0})]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual([r["backend_id"] for r in out["results"]], ["b0", "b1"])

    def test_prefers_smaller_ratio(self):
        # 第一批后 b0=1/2、b1=1/3；第三个作业应选比值更小的 b1（1/3 < 1/2）。
        backends = [
            backend("b0", slots=2, representations=("state_vector",)),
            backend("b1", slots=3, representations=("state_vector",)),
            backend("bx", slots=1, representations=("density_matrix",)),
        ]
        jobs = [
            job("a", values={"theta": 0.0}),
            job("b", values={"theta": 0.0},
                noise={"single_qubit_depolarizing": 0.01}),
            job("c", values={"theta": 0.0}),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual([r["backend_id"] for r in out["results"]], ["b0", "bx", "b1"])
        self.assertEqual(
            [(b["id"], b["assigned_count"]) for b in out["backends"]],
            [("b0", 1), ("b1", 1), ("bx", 1)],
        )

    def test_cross_multiply_equal_ratios_earlier_wins(self):
        # 2/4 与 1/2 相等：已分到 2 个的 b0(slots=4) 不应抢在 1/2 的 b1 之前？
        # 构造 b0=1/2、b1=2/4：相等取靠前 b0。
        backends = [
            backend("b0", slots=2, representations=("state_vector",)),
            backend("b1", slots=4, representations=("state_vector",)),
        ]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(4)]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        picks = [r["backend_id"] for r in out["results"]]
        # j0: b0(0/2), j1: b1(0/4), j2: 比值 1/2 vs 1/4 -> b1,
        # j3: b0=1/2, b1=2/4 相等 -> b0。
        self.assertEqual(picks, ["b0", "b1", "b1", "b0"])


class IneligibilityReasonsTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_unsupported_representation(self):
        backends = [backend("sv", representations=("state_vector",))]
        jobs = [job("a", values={"theta": 0.0},
                    noise={"single_qubit_depolarizing": 0.2})]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        item = out["results"][0]
        self.assertEqual(item["status"], "no_eligible_backend")
        self.assertIs(item["assigned"], False)
        self.assertIsNone(item["backend_id"])
        self.assertIsNone(item["requirements"])
        self.assertIsNone(item["validation_error"])
        self.assertEqual(item["reasons"], [
            {"backend_id": "sv", "reasons": ["unsupported_representation"]},
        ])
        self.assertEqual(out["summary"], {"total": 1, "assigned": 0, "rejected": 1})

    def test_no_slot_once_filled(self):
        backends = [backend("only", slots=1)]
        jobs = [job("a", values={"theta": 0.0}), job("b", values={"theta": 0.0})]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(out["results"][0]["status"], "assigned")
        second = out["results"][1]
        self.assertEqual(second["status"], "no_eligible_backend")
        self.assertEqual(second["reasons"], [
            {"backend_id": "only", "reasons": ["no_slot"]},
        ])

    def test_each_budget_key_reported(self):
        cases = [
            ("max_state_bytes", {"max_state_bytes": 8}),
            ("max_circuit_evaluations", {"max_circuit_evaluations": 0}),
            ("max_gate_applications", {"max_gate_applications": 2}),
            ("max_total_shots", {"max_total_shots": 9}),
        ]
        for key, overrides in cases:
            with self.subTest(key=key):
                backends = [backend("b", **overrides)]
                jobs = [job("a", ["ZZ"], values={"theta": 0.0}, shots=10)]
                out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
                item = out["results"][0]
                self.assertEqual(item["status"], "no_eligible_backend")
                self.assertEqual(item["reasons"][0]["reasons"], [key])

    def test_budget_boundary_equal_is_eligible(self):
        backends = [backend(
            "b", max_state_bytes=64, max_circuit_evaluations=1,
            max_gate_applications=3, max_total_shots=20,
        )]
        jobs = [job("a", ["ZZ"], values={"theta": 0.0}, shots=10)]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(out["results"][0]["status"], "assigned")

    def test_reasons_follow_backend_order_and_accumulate(self):
        # b0 表示不支持；b1 满 slot 且预算超限；两个原因都应列出。
        backends = [
            backend("b0", representations=("state_vector",)),
            backend("b1", slots=1, max_total_shots=1),
        ]
        # 先用一个噪声作业填满 b1 的唯一 slot。
        jobs = [
            job("fill", values={"theta": 0.0},
                noise={"single_qubit_depolarizing": 0.1}),
            job("a", ["ZZ"], values={"theta": 0.0}, shots=10,
                noise={"single_qubit_depolarizing": 0.1}),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(out["results"][0]["backend_id"], "b1")
        item = out["results"][1]
        self.assertEqual(item["status"], "no_eligible_backend")
        self.assertEqual([r["backend_id"] for r in item["reasons"]], ["b0", "b1"])
        self.assertEqual(item["reasons"][0]["reasons"], ["unsupported_representation"])
        self.assertEqual(
            item["reasons"][1]["reasons"], ["no_slot", "max_total_shots"],
        )


class JobRejectionTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("b0")]

    def _plan_one(self, job_spec):
        return self.svc.plan_batch_offload(CIRCUIT, [job_spec], self.backends)["results"][0]

    def test_invalid_observables_rejected_isolated(self):
        jobs = [
            job("bad", ["ZZZ"], values={"theta": 0.0}),
            job("good", values={"theta": 0.0}),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, self.backends)
        bad, good = out["results"]
        self.assertEqual(bad["status"], "rejected")
        self.assertIs(bad["assigned"], False)
        self.assertIsNone(bad["backend_id"])
        self.assertIsNone(bad["requirements"])
        self.assertIsNone(bad["reasons"])
        self.assertEqual(bad["validation_error"], {
            "type": "SimulationError",
            "code": "invalid_observable",
            "path": "observables[0]",
            "message": "observable at observables[0] must have length 2",
        })
        self.assertEqual(good["status"], "assigned")
        self.assertEqual(out["summary"], {"total": 2, "assigned": 1, "rejected": 1})

    def test_invalid_shots_bool_and_zero(self):
        for bad_shots in (True, 0, -3, 1.5):
            with self.subTest(bad_shots=bad_shots):
                item = self._plan_one(job("a", values={"theta": 0.0}, shots=bad_shots))
                self.assertEqual(item["status"], "rejected")
                self.assertEqual(item["validation_error"]["code"], "invalid_shots")
                self.assertEqual(item["validation_error"]["path"], "shots")
                self.assertEqual(item["validation_error"]["type"], "SimulationError")

    def test_seed_without_shots_rejected(self):
        item = self._plan_one(job("a", values={"theta": 0.0}, seed=3))
        self.assertEqual(item["status"], "rejected")
        self.assertEqual(item["validation_error"]["code"], "seed_without_shots")

    def test_missing_binding_rejected(self):
        item = self._plan_one(job("a"))
        self.assertEqual(item["status"], "rejected")
        self.assertEqual(item["validation_error"]["type"], "ParameterBindingError")
        self.assertEqual(item["validation_error"]["code"], "missing_parameter")

    def test_unknown_binding_rejected(self):
        item = self._plan_one(job("a", values={"theta": 0.0, "extra": 1.0}))
        self.assertEqual(item["validation_error"]["code"], "unknown_parameter")

    def test_rejected_jobs_do_not_consume_slots(self):
        backends = [backend("only", slots=1)]
        jobs = [
            job("bad", ["ZZZ"], values={"theta": 0.0}),
            job("good", values={"theta": 0.0}),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(out["results"][1]["status"], "assigned")
        self.assertEqual(out["backends"][0]["assigned_count"], 1)

    def test_runtime_qubit_limit_is_not_a_validation_error(self):
        # 超过 20/10 量子位运行时上限在资源口径中只是需求画像（state_bytes
        # 巨大），作业本身仍然合法；能否分配完全由后端预算决定。
        big = {"qubit_count": 21, "parameters": [], "operations": []}
        big_backend = backend(
            "big", representations=("state_vector",),
            max_state_bytes=2 ** 21 * 16,
        )
        out = self.svc.plan_batch_offload(big, [job("a", ["I" * 21])], [big_backend])
        item = out["results"][0]
        self.assertEqual(item["status"], "assigned")
        self.assertEqual(item["requirements"]["state_elements"], 1 << 21)

        small = [backend("small", representations=("state_vector",),
                         max_state_bytes=1024)]
        out2 = self.svc.plan_batch_offload(big, [job("a", ["I" * 21])], small)
        self.assertEqual(out2["results"][0]["status"], "no_eligible_backend")
        self.assertEqual(
            out2["results"][0]["reasons"][0]["reasons"], ["max_state_bytes"],
        )


class BackendValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.jobs = [job("a", values={"theta": 0.0})]

    def _expect(self, backends, code, path):
        with self.assertRaises(OffloadPlanningError) as ctx:
            self.svc.plan_batch_offload(CIRCUIT, self.jobs, backends)
        self.assertEqual(ctx.exception.code, code)
        self.assertEqual(ctx.exception.path, path)

    def test_not_array_or_empty(self):
        self._expect({"id": "x"}, "invalid_backends", "backends")
        self._expect([], "invalid_backends", "backends")
        self._expect(None, "invalid_backends", "backends")

    def test_backend_not_object(self):
        self._expect(["x"], "invalid_backend", "backends[0]")

    def test_unknown_field(self):
        bad = backend("b0")
        bad["extra"] = 1
        self._expect([bad], "invalid_backend", "backends[0].extra")

    def test_bad_id(self):
        self._expect([backend("")], "invalid_backend", "backends[0].id")
        self._expect([backend(1)], "invalid_backend", "backends[0].id")
        missing = backend("b0")
        del missing["id"]
        self._expect([missing], "invalid_backend", "backends[0].id")

    def test_bad_representations(self):
        cases = [
            (None, "backends[0].representations"),
            ([], "backends[0].representations"),
            ("state_vector", "backends[0].representations"),
            (["state_vector", "state_vector"], "backends[0].representations[1]"),
            (["matrix"], "backends[0].representations[0]"),
        ]
        for reps, path in cases:
            with self.subTest(reps=reps):
                bad = backend("b0")
                bad["representations"] = reps
                self._expect([bad], "invalid_backend", path)

    def test_bad_budget_values(self):
        for key in ("max_state_bytes", "max_circuit_evaluations",
                    "max_gate_applications", "max_total_shots"):
            for value in (False, -1, 1.0, "10"):
                with self.subTest(key=key, value=value):
                    bad = backend("b0")
                    bad[key] = value
                    self._expect([bad], "invalid_backend", f"backends[0].{key}")
            with self.subTest(key=key, value="missing"):
                bad = backend("b0")
                del bad[key]
                self._expect([bad], "invalid_backend", f"backends[0].{key}")

    def test_bad_slots(self):
        for value in (0, -1, True, 1.0, None):
            with self.subTest(value=value):
                bad = backend("b0")
                bad["slots"] = value
                self._expect([bad], "invalid_backend", "backends[0].slots")

    def test_duplicate_backend_id(self):
        self._expect([backend("dup"), backend("dup")],
                     "duplicate_backend_id", "backends[1].id")


class JobsStructureTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.backends = [backend("b0")]

    def test_jobs_errors_are_batch_execution_error(self):
        for bad_jobs, code, path in [
            ([], "invalid_batch", "jobs"),
            ("x", "invalid_batch", "jobs"),
            ([{"observables": ["ZZ"]}], "invalid_job", "jobs[0].id"),
            ([{"id": "a"}], "invalid_job", "jobs[0].observables"),
            ([{"id": "a", "observables": ["ZZ"], "bogus": 1}],
             "invalid_job", "jobs[0].bogus"),
            ([{"id": "a", "observables": ["ZZ"]},
              {"id": "a", "observables": ["ZZ"]}],
             "duplicate_job_id", "jobs[1].id"),
        ]:
            with self.subTest(code=code, path=path):
                with self.assertRaises(BatchExecutionError) as ctx:
                    self.svc.plan_batch_offload(CIRCUIT, bad_jobs, self.backends)
                self.assertEqual(ctx.exception.code, code)
                self.assertEqual(ctx.exception.path, path)


class ValidationOrderTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_circuit_before_backends_before_jobs(self):
        bad_circuit = {"qubit_count": -1, "parameters": [], "operations": []}
        with self.assertRaises(CircuitValidationError):
            self.svc.plan_batch_offload(bad_circuit, [], [])

        good_jobs = [job("a", values={"theta": 0.0})]
        with self.assertRaises(OffloadPlanningError):
            self.svc.plan_batch_offload(CIRCUIT, good_jobs, [])

        # backends 合法但 jobs 结构错误时才轮到 BatchExecutionError。
        with self.assertRaises(BatchExecutionError):
            self.svc.plan_batch_offload(CIRCUIT, [], [backend("b0")])

    def test_no_partial_plan_on_request_failure(self):
        # 请求级失败以异常形式表现，没有任何返回对象。
        with self.assertRaises(OffloadPlanningError):
            self.svc.plan_batch_offload(CIRCUIT, [job("a")], None)


class ErrorExportTest(unittest.TestCase):
    def test_inheritance_and_identity(self):
        self.assertTrue(issubclass(OffloadPlanningError, ValueError))
        self.assertIs(RootOffloadPlanningError, OffloadPlanningError)
        exc = OffloadPlanningError("invalid_backends", "backends")
        self.assertEqual(exc.code, "invalid_backends")
        self.assertEqual(exc.path, "backends")
        self.assertIn("invalid_backends", str(exc))


if __name__ == "__main__":
    unittest.main()
