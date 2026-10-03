import copy
import json
import unittest

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
        {"gate": "rx", "target": 1, "angle": {"parameter": "theta"}},
        {"gate": "cx", "control": 0, "target": 1},
    ],
}


class BatchExpectationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_success_matches_single_expectation_exact(self):
        jobs = [
            {"id": "a", "observables": ["ZI", "IZ"], "values": {"theta": 0.3}},
            {"id": "b", "observables": ["XX"], "values": {"theta": -1.2}},
        ]
        outcome = self.svc.expectation_batch(CIRCUIT, jobs)
        self.assertEqual([item["id"] for item in outcome["results"]], ["a", "b"])
        for item, job in zip(outcome["results"], jobs):
            self.assertEqual(item["status"], "succeeded")
            self.assertIsNone(item["error"])
            single = self.svc.expectation(
                CIRCUIT, job["observables"], values=job["values"],
            )
            self.assertEqual(item["result"], single)
        self.assertEqual(outcome["summary"], {"total": 2, "succeeded": 2, "failed": 0})

    def test_sampled_counts_and_seed_reproducible(self):
        jobs = [
            {"id": "s1", "observables": ["ZZ", "XI"], "values": {"theta": 0.7},
             "shots": 200, "seed": 42},
            {"id": "s2", "observables": ["YY"], "values": {"theta": 0.7},
             "shots": 150, "seed": 7,
             "noise": {"single_qubit_depolarizing": 0.1}},
        ]
        outcome = self.svc.expectation_batch(CIRCUIT, jobs, max_concurrency=2)
        for item, job in zip(outcome["results"], jobs):
            single = self.svc.expectation(
                CIRCUIT, job["observables"], values=job["values"],
                shots=job["shots"], seed=job["seed"], noise=job.get("noise"),
            )
            self.assertEqual(item["result"], single)
        again = self.svc.expectation_batch(CIRCUIT, jobs, max_concurrency=2)
        self.assertEqual(outcome, again)

    def test_results_identical_across_concurrency_levels(self):
        jobs = [
            {"id": f"j{i}", "observables": ["ZI", "IZ", "XX"],
             "values": {"theta": 0.1 * i}, "shots": 100, "seed": i}
            for i in range(1, 8)
        ]
        baseline = self.svc.expectation_batch(CIRCUIT, jobs)
        for workers in (2, 3, 8, 100):
            outcome = self.svc.expectation_batch(CIRCUIT, jobs, max_concurrency=workers)
            self.assertEqual(outcome, baseline)

    def test_failed_job_isolated_and_reports_error(self):
        jobs = [
            {"id": "ok", "observables": ["ZI"], "values": {"theta": 0.5}},
            {"id": "bad-bind", "observables": ["ZI"], "values": {}},
            {"id": "bad-sim", "observables": ["Z"], "values": {"theta": 0.5}},
            {"id": "ok2", "observables": ["IZ"], "values": {"theta": 0.5}},
        ]
        outcome = self.svc.expectation_batch(CIRCUIT, jobs, max_concurrency=3)
        results = outcome["results"]
        self.assertEqual([item["status"] for item in results],
                         ["succeeded", "failed", "failed", "succeeded"])

        bind_error = results[1]["error"]
        self.assertEqual(results[1]["result"], {})
        self.assertEqual(bind_error["type"], "ParameterBindingError")
        self.assertEqual(bind_error["code"], "missing_parameter")
        self.assertEqual(bind_error["path"], "theta")

        sim_error = results[2]["error"]
        self.assertEqual(sim_error["type"], "SimulationError")
        self.assertEqual(sim_error["code"], "invalid_observable")
        self.assertEqual(sim_error["path"], "observables[0]")

        self.assertEqual(outcome["summary"], {"total": 4, "succeeded": 2, "failed": 2})
        json.dumps(outcome, sort_keys=True)

    def test_results_follow_input_order_not_completion_order(self):
        jobs = [
            {"id": f"job-{i}", "observables": ["ZI"], "values": {"theta": 0.01 * i}}
            for i in range(10)
        ]
        outcome = self.svc.expectation_batch(CIRCUIT, jobs, max_concurrency=4)
        self.assertEqual([item["id"] for item in outcome["results"]],
                         [f"job-{i}" for i in range(10)])

    def test_inputs_are_not_mutated(self):
        jobs = [
            {"id": "a", "observables": ["ZI"], "values": {"theta": 0.3},
             "shots": 50, "seed": 1,
             "noise": {"single_qubit_depolarizing": 0.2}},
            {"id": "b", "observables": ["IZ"], "values": {}},
        ]
        circuit_snapshot = copy.deepcopy(CIRCUIT)
        jobs_snapshot = copy.deepcopy(jobs)
        self.svc.expectation_batch(CIRCUIT, jobs, max_concurrency=2)
        self.assertEqual(CIRCUIT, circuit_snapshot)
        self.assertEqual(jobs, jobs_snapshot)

    def test_circuit_validation_error_still_raised(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.expectation_batch(
                {"qubit_count": 1, "operations": [{"gate": "nope", "target": 0}]},
                [{"id": "a", "observables": ["Z"]}],
            )

    def test_circuit_validated_before_jobs(self):
        exc = None
        try:
            self.svc.expectation_batch({"qubit_count": "two"}, "not-a-list")
        except CircuitValidationError as err:
            exc = err
        self.assertIsNotNone(exc)

    def test_concurrency_validated_before_jobs(self):
        err = batch_err(self.svc.expectation_batch, CIRCUIT, "not-a-list",
                        max_concurrency=0)
        self.assertEqual(err.code, "invalid_concurrency")

    def test_invalid_batch(self):
        for jobs in (None, "jobs", {}, 3, []):
            err = batch_err(self.svc.expectation_batch, CIRCUIT, jobs)
            self.assertEqual(err.code, "invalid_batch")
            self.assertEqual(err.path, "jobs")

    def test_invalid_concurrency(self):
        jobs = [{"id": "a", "observables": ["ZI"], "values": {"theta": 1.0}}]
        for bad in (0, -1, 1.5, True, "2"):
            err = batch_err(self.svc.expectation_batch, CIRCUIT, jobs,
                            max_concurrency=bad)
            self.assertEqual(err.code, "invalid_concurrency")
            self.assertEqual(err.path, "max_concurrency")

    def test_invalid_job_cases(self):
        cases = [
            (["nope"], "jobs[0]"),
            ([{"observables": ["ZI"]}], "jobs[0].id"),
            ([{"id": 1, "observables": ["ZI"]}], "jobs[0].id"),
            ([{"id": "", "observables": ["ZI"]}], "jobs[0].id"),
            ([{"id": "a"}], "jobs[0].observables"),
            ([{"id": "a", "observables": ["ZI"], "extra": 1}], "jobs[0].extra"),
        ]
        for jobs, path in cases:
            err = batch_err(self.svc.expectation_batch, CIRCUIT, jobs)
            self.assertEqual(err.code, "invalid_job")
            self.assertEqual(err.path, path)

    def test_duplicate_job_id_points_to_later_occurrence(self):
        jobs = [
            {"id": "a", "observables": ["ZI"]},
            {"id": "b", "observables": ["ZI"]},
            {"id": "a", "observables": ["IZ"]},
        ]
        err = batch_err(self.svc.expectation_batch, CIRCUIT, jobs)
        self.assertEqual(err.code, "duplicate_job_id")
        self.assertEqual(err.path, "jobs[2].id")

    def test_default_concurrency_is_sequential_and_matches(self):
        jobs = [
            {"id": "x", "observables": ["ZI"], "values": {"theta": 0.4},
             "shots": 64, "seed": 9},
        ]
        default = self.svc.expectation_batch(CIRCUIT, jobs)
        explicit = self.svc.expectation_batch(CIRCUIT, jobs, max_concurrency=1)
        self.assertEqual(default, explicit)


if __name__ == "__main__":
    unittest.main()
