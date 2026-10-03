import copy
import json
import unittest

from qubitfabric.service import (
    BatchExecutionError,
    CircuitValidationError,
    OffloadPlanningError,
    Service,
)
from qubitfabric import OffloadPlanningError as PkgOffloadError

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


def plan_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except OffloadPlanningError as exc:
        return exc
    raise AssertionError("OffloadPlanningError not raised")


class AssignmentTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_assigned_shape_and_requirements_match_resource_estimator(self):
        backends = [backend("a"), backend("b")]
        jobs = [
            job("exact", values={"theta": 0.0}),
            job("sampled", values={"theta": 0.0}, shots=50, seed=3),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual([item["id"] for item in out["results"]], ["exact", "sampled"])
        for item in out["results"]:
            self.assertEqual(item["status"], "assigned")
            self.assertEqual(set(item), {"id", "status", "backend_id", "requirements"})
            self.assertEqual(
                set(item["requirements"]),
                {
                    "representation", "state_bytes", "circuit_evaluations",
                    "gate_applications", "total_shots",
                },
            )

        specs = [
            ("exact_expectation", jobs[0]),
            ("sampled_expectation", jobs[1]),
        ]
        for item, (req_type, spec) in zip(out["results"], specs):
            request = {
                "type": req_type,
                "observables": spec["observables"],
                "values": spec.get("values"),
                "noise": spec.get("noise"),
            }
            if "shots" in spec:
                request["shots"] = spec["shots"]
                request["seed"] = spec["seed"]
            resources = self.svc.estimate_resources(CIRCUIT, request)
            req = item["requirements"]
            self.assertEqual(req["representation"], resources["representation"])
            self.assertEqual(req["state_bytes"], resources["state_bytes"])
            self.assertEqual(req["circuit_evaluations"], resources["circuit_evaluations"])
            self.assertEqual(req["gate_applications"], resources["gate_applications"])
            self.assertEqual(req["total_shots"], resources["total_shots"])

        self.assertEqual(
            out["summary"],
            {
                "backend_assignments": [
                    {"backend_id": "a", "assigned": 1},
                    {"backend_id": "b", "assigned": 1},
                ],
                "total": 2,
                "assigned": 2,
                "rejected": 0,
            },
        )

    def test_ratio_balancing_with_cross_multiplication_and_input_order_tie(self):
        # slots 2 vs 3：比例 0 相等时取靠前的 a；随后按 assigned/slots
        # 最小者分配（交叉乘法，无浮点）。
        backends = [backend("a", slots=2), backend("b", slots=3)]
        jobs = [job(f"j{i}", values={"theta": 0.0}) for i in range(6)]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(
            [item["backend_id"] for item in out["results"][:5]],
            ["a", "b", "b", "a", "b"],
        )
        self.assertEqual(out["results"][5]["status"], "no_eligible_backend")
        self.assertEqual(
            out["summary"]["backend_assignments"],
            [{"backend_id": "a", "assigned": 2}, {"backend_id": "b", "assigned": 3}],
        )
        self.assertEqual(out["summary"], {
            "backend_assignments": [
                {"backend_id": "a", "assigned": 2},
                {"backend_id": "b", "assigned": 3},
            ],
            "total": 6, "assigned": 5, "rejected": 1,
        })

    def test_equal_ratio_prefers_earlier_backend(self):
        backends = [backend("a", slots=1), backend("b", slots=1)]
        out = self.svc.plan_batch_offload(CIRCUIT, [job("j", values={"theta": 0.0})], backends)
        self.assertEqual(out["results"][0]["backend_id"], "a")

    def test_noise_selects_density_matrix_representation(self):
        noisy = job("n", values={"theta": 0.0},
                    noise={"single_qubit_depolarizing": 0.1})
        # 仅支持 state_vector 的后端不可用，密度矩阵后端接住作业。
        backends = [
            backend("sv", reps=["state_vector"]),
            backend("dm", reps=["density_matrix"]),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, [noisy], backends)
        item = out["results"][0]
        self.assertEqual(item["status"], "assigned")
        self.assertEqual(item["backend_id"], "dm")
        self.assertEqual(item["requirements"]["representation"], "density_matrix")
        # 2 量子位密度矩阵 4^2 元素 * 16 字节 = 256。
        self.assertEqual(item["requirements"]["state_bytes"], 256)

    def test_requirement_equal_to_limit_is_admitted(self):
        b = backend("a", slots=1, max_state_bytes=64, max_circuit_evaluations=1,
                    max_gate_applications=3, max_total_shots=10)
        j = job("j", ["ZZ"], values={"theta": 0.0}, shots=10)
        out = self.svc.plan_batch_offload(CIRCUIT, [j], [b])
        self.assertEqual(out["results"][0]["status"], "assigned")

    def test_sampled_total_shots_scales_with_observable_count(self):
        b = backend("a", slots=1)
        j = job("j", ["ZZ", "XX", "ZI"], values={"theta": 0.0}, shots=7)
        out = self.svc.plan_batch_offload(CIRCUIT, [j], [b])
        self.assertEqual(out["results"][0]["requirements"]["total_shots"], 21)
        # 精确模式总 shots 恒为 0。
        j_exact = job("je", ["ZZ"], values={"theta": 0.0})
        out = self.svc.plan_batch_offload(CIRCUIT, [j_exact], [b])
        self.assertEqual(out["results"][0]["requirements"]["total_shots"], 0)


class NoEligibleBackendTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_reasons_follow_backend_order_and_budget_key_order(self):
        # a：表示不支持且预算多项超限；b：slot 用尽。
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
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, [a, b])
        self.assertEqual(out["results"][0]["status"], "assigned")
        self.assertEqual(out["results"][0]["backend_id"], "b")
        item = out["results"][1]
        self.assertEqual(item["status"], "no_eligible_backend")
        self.assertIsNone(item["requirements"])
        self.assertEqual(set(item), {"id", "status", "requirements", "reasons"})
        self.assertEqual(
            item["reasons"],
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
        self.assertEqual(out["summary"]["rejected"], 1)

    def test_total_shots_budget_exceeded_only_for_sampled_jobs(self):
        tight = backend("a", slots=2, max_total_shots=99)
        jobs = [
            job("exact", ["ZZ"], values={"theta": 0.0}),
            job("sampled", ["ZZ", "XX"], values={"theta": 0.0}, shots=50),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, [tight])
        self.assertEqual(out["results"][0]["status"], "assigned")
        item = out["results"][1]
        self.assertEqual(item["status"], "no_eligible_backend")
        self.assertEqual(item["reasons"], [["max_total_shots"]])

    def test_all_backends_full_reports_no_slot_per_backend(self):
        out = self.svc.plan_batch_offload(
            CIRCUIT,
            [job("j", values={"theta": 0.0}), job("k", values={"theta": 0.0}),
             job("m", values={"theta": 0.0})],
            [backend("a", slots=1), backend("b", slots=1)],
        )
        last = out["results"][2]
        self.assertEqual(last["status"], "no_eligible_backend")
        self.assertEqual(last["reasons"], [["no_slot"], ["no_slot"]])


class RejectionIsolationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_semantically_invalid_jobs_are_rejected_but_others_continue(self):
        jobs = [
            job("ok", values={"theta": 0.0}),
            job("missing-binding"),
            job("bad-observable", ["ZZZ"], values={"theta": 0.0}),
            job("bad-shots", values={"theta": 0.0}, shots=0),
            job("seed-no-shots", values={"theta": 0.0}, seed=3),
            job("ok2", values={"theta": 1.0}, shots=10, seed=1),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, [backend("a", slots=10)])
        self.assertEqual([item["id"] for item in out["results"]],
                         ["ok", "missing-binding", "bad-observable",
                          "bad-shots", "seed-no-shots", "ok2"])
        statuses = [item["status"] for item in out["results"]]
        self.assertEqual(
            statuses,
            ["assigned", "rejected", "rejected", "rejected", "rejected", "assigned"],
        )

        expected = [
            ("ParameterBindingError", "missing_parameter", "theta"),
            ("SimulationError", "invalid_observable", "observables[0]"),
            ("SimulationError", "invalid_shots", "shots"),
            ("SimulationError", "seed_without_shots", "seed"),
        ]
        for item, (etype, code, path) in zip(out["results"][1:5], expected):
            self.assertIsNone(item["requirements"])
            self.assertEqual(set(item), {"id", "status", "requirements", "validation_error"})
            err = item["validation_error"]
            self.assertEqual(set(err), {"type", "code", "path", "message"})
            self.assertEqual((err["type"], err["code"], err["path"]), (etype, code, path))
            self.assertIsInstance(err["message"], str)
            self.assertTrue(err["message"])

        self.assertEqual(
            out["summary"],
            {
                "backend_assignments": [{"backend_id": "a", "assigned": 2}],
                "total": 6, "assigned": 2, "rejected": 4,
            },
        )

    def test_rejected_job_does_not_consume_slot(self):
        jobs = [
            job("bad"),
            job("ok", values={"theta": 0.0}),
        ]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, [backend("a", slots=1)])
        self.assertEqual(out["results"][0]["status"], "rejected")
        self.assertEqual(out["results"][1]["status"], "assigned")
        self.assertEqual(out["summary"]["backend_assignments"],
                         [{"backend_id": "a", "assigned": 1}])

    def test_state_space_too_large_is_rejected(self):
        out = self.svc.plan_batch_offload(
            {"qubit_count": 21},
            [job("big", ["I" * 21])],
            [backend("a")],
        )
        item = out["results"][0]
        self.assertEqual(item["status"], "rejected")
        self.assertEqual(item["validation_error"]["type"], "SimulationError")
        self.assertEqual(item["validation_error"]["code"], "state_space_too_large")
        self.assertEqual(item["validation_error"]["path"], "qubit_count")

    def test_validation_error_message_matches_direct_entry(self):
        jobs = [job("bad", ["ZZZ"], values={"theta": 0.0})]
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, [backend("a")])
        try:
            self.svc.expectation(CIRCUIT, ["ZZZ"], values={"theta": 0.0})
        except Exception as exc:  # noqa: BLE001 - 对比同一异常文本
            direct = str(exc)
        self.assertEqual(out["results"][0]["validation_error"]["message"], direct)


class RequestValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.good_jobs = [job("j", values={"theta": 0.0})]

    def test_circuit_error_has_priority_over_everything(self):
        with self.assertRaises(CircuitValidationError) as ctx:
            self.svc.plan_batch_offload({"qubit_count": -1}, None, None)
        self.assertEqual(ctx.exception.code, "invalid_value")
        with self.assertRaises(CircuitValidationError):
            self.svc.plan_batch_offload("nope", [], [])

    def test_backends_must_be_non_empty_array(self):
        for bad in (None, [], {}, "backends", 1, True):
            exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, bad)
            self.assertEqual((exc.code, exc.path), ("invalid_backends", "backends"), bad)

    def test_backend_must_be_object(self):
        exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, ["x"])
        self.assertEqual((exc.code, exc.path), ("invalid_backend", "backends[0]"))

    def test_unknown_backend_field_rejected(self):
        b = backend("a")
        b["bogus"] = 1
        exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, [b])
        self.assertEqual((exc.code, exc.path), ("invalid_backend", "backends[0].bogus"))

    def test_id_required_non_empty_string(self):
        for mutate, path in (
            (lambda b: b.pop("id"), "backends[0].id"),
            (lambda b: b.update(id=4), "backends[0].id"),
            (lambda b: b.update(id=""), "backends[0].id"),
            (lambda b: b.update(id=None), "backends[0].id"),
        ):
            b = backend("a")
            mutate(b)
            exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, [b])
            self.assertEqual((exc.code, exc.path), ("invalid_backend", path))

    def test_representations_validation(self):
        cases = [
            (lambda b: b.pop("representations"), "backends[0].representations"),
            (lambda b: b.update(representations=[]), "backends[0].representations"),
            (lambda b: b.update(representations="state_vector"),
             "backends[0].representations"),
            (lambda b: b.update(representations=["bogus"]),
             "backends[0].representations[0]"),
            (lambda b: b.update(representations=["state_vector", "state_vector"]),
             "backends[0].representations[1]"),
        ]
        for mutate, path in cases:
            b = backend("a")
            mutate(b)
            exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, [b])
            self.assertEqual((exc.code, exc.path), ("invalid_backend", path), path)

    def test_budget_fields_must_be_non_negative_ints_excluding_bool(self):
        for key in ("max_state_bytes", "max_circuit_evaluations",
                    "max_gate_applications", "max_total_shots"):
            for bad in (-1, 1.5, "1", True, False):
                b = backend("a")
                b[key] = bad
                exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, [b])
                self.assertEqual(
                    (exc.code, exc.path), ("invalid_backend", f"backends[0].{key}"),
                    (key, bad),
                )
            missing = backend("a")
            missing.pop(key)
            exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, [missing])
            self.assertEqual(
                (exc.code, exc.path), ("invalid_backend", f"backends[0].{key}"),
            )

    def test_zero_budgets_are_valid(self):
        b = backend("a", max_state_bytes=0, max_circuit_evaluations=0,
                    max_gate_applications=0, max_total_shots=0)
        out = self.svc.plan_batch_offload(CIRCUIT, self.good_jobs, [b])
        self.assertEqual(out["results"][0]["status"], "no_eligible_backend")

    def test_slots_must_be_positive_int_excluding_bool(self):
        for bad in (0, -1, 1.5, "2", True, False):
            b = backend("a", slots=bad)
            exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, [b])
            self.assertEqual(
                (exc.code, exc.path), ("invalid_backend", "backends[0].slots"), bad,
            )
        missing = backend("a")
        missing.pop("slots")
        exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, self.good_jobs, [missing])
        self.assertEqual((exc.code, exc.path), ("invalid_backend", "backends[0].slots"))

    def test_duplicate_backend_id_points_to_later_occurrence(self):
        exc = plan_err(
            self.svc.plan_batch_offload, CIRCUIT, self.good_jobs,
            [backend("x"), backend("y"), backend("x")],
        )
        self.assertEqual((exc.code, exc.path), ("duplicate_backend_id", "backends[2].id"))

    def test_jobs_structure_errors_raise_batch_execution_error(self):
        for bad in (None, [], {}, 1):
            with self.assertRaises(BatchExecutionError) as ctx:
                self.svc.plan_batch_offload(CIRCUIT, bad, [backend("a")])
            self.assertEqual(ctx.exception.code, "invalid_batch")
        with self.assertRaises(BatchExecutionError) as ctx:
            self.svc.plan_batch_offload(
                CIRCUIT, [{"observables": ["ZZ"]}], [backend("a")],
            )
        self.assertEqual((ctx.exception.code, ctx.exception.path),
                         ("invalid_job", "jobs[0].id"))
        with self.assertRaises(BatchExecutionError) as ctx:
            self.svc.plan_batch_offload(
                CIRCUIT,
                [job("x", values={"theta": 0.0}), job("x", values={"theta": 0.0})],
                [backend("a")],
            )
        self.assertEqual(ctx.exception.code, "duplicate_job_id")

    def test_validation_order_circuit_backends_jobs(self):
        # 电路正常时，backends 错误优先于 jobs 结构错误。
        exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, None, None)
        self.assertEqual(exc.code, "invalid_backends")
        exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, [], [])
        self.assertEqual(exc.code, "invalid_backends")
        # backends 与 jobs 同时非法时，先报 backends。
        exc = plan_err(self.svc.plan_batch_offload, CIRCUIT, "not-jobs", backend("a"))
        self.assertEqual(exc.code, "invalid_backends")
        # backends 合法后才轮到 jobs 结构错误。
        with self.assertRaises(BatchExecutionError):
            self.svc.plan_batch_offload(CIRCUIT, "not-jobs", [backend("a")])

    def test_no_partial_plan_on_request_failure(self):
        # 请求级失败直接抛异常，不返回任何规划结果。
        with self.assertRaises((CircuitValidationError, OffloadPlanningError, BatchExecutionError)):
            self.svc.plan_batch_offload({"qubit_count": -1}, self.good_jobs, [backend("a")])


class OffloadErrorHierarchyTest(unittest.TestCase):
    def test_is_value_error_with_code_and_path(self):
        exc = OffloadPlanningError("invalid_backends", "backends")
        self.assertIsInstance(exc, ValueError)
        self.assertEqual(exc.code, "invalid_backends")
        self.assertEqual(exc.path, "backends")
        self.assertIn("invalid_backends", str(exc))

    def test_exported_from_package_and_service(self):
        self.assertIs(PkgOffloadError, OffloadPlanningError)


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
        out = self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        json.dumps(out)

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
        self.svc.plan_batch_offload(CIRCUIT, jobs, backends)
        self.assertEqual(CIRCUIT, circuit_copy)
        self.assertEqual(jobs, jobs_copy)
        self.assertEqual(backends, backends_copy)

    def test_planning_does_not_execute_simulation(self):
        # 只规划：结果不含任何期望值/计数字段。
        out = self.svc.plan_batch_offload(
            CIRCUIT,
            [job("a", values={"theta": 0.0}, shots=10, seed=1)],
            [backend("x")],
        )
        item = out["results"][0]
        self.assertNotIn("result", item)
        self.assertNotIn("expectation", item["requirements"])
        self.assertNotIn("counts", item["requirements"])


if __name__ == "__main__":
    unittest.main()
