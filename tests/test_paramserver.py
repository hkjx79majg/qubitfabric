import copy
import hashlib
import json
import unittest

from qubitfabric import ParameterServerError as PackageParameterServerError
from qubitfabric.service import (
    CircuitValidationError,
    ParameterBindingError,
    ParameterServerError,
    Service,
)


def rx_circuit():
    return {"qubit_count": 1, "parameters": ["theta"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
    ]}


def two_param_circuit():
    return {"qubit_count": 2, "parameters": ["theta", "phi"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        {"gate": "rz", "target": 1, "angle": {"parameter": "phi"}},
    ]}


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(update_id, base_revision, gradients):
    payload = {"id": update_id, "base_revision": base_revision, "gradients": gradients}
    return hashlib.sha256(canon(payload).encode("utf-8")).hexdigest()


def update(update_id, base_revision, gradients):
    return {"id": update_id, "base_revision": base_revision, "gradients": gradients}


def ps_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ParameterServerError as exc:
        return exc
    raise AssertionError("ParameterServerError not raised")


class CreateParameterStateTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_state_shape_and_types(self):
        state = self.svc.create_parameter_state(rx_circuit(), {"theta": 0.3})
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["parameters"], ["theta"])
        self.assertEqual(state["revision"], 0)
        self.assertEqual(state["values"], {"theta": 0.3})
        self.assertIsInstance(state["values"]["theta"], float)
        self.assertEqual(state["updates"], [])
        self.assertEqual(set(state), {"version", "parameters", "revision", "values", "updates"})

    def test_state_is_json_round_trippable(self):
        state = self.svc.create_parameter_state(two_param_circuit(), {"theta": 1, "phi": -2.5})
        clone = json.loads(json.dumps(state))
        self.assertEqual(state, clone)
        self.assertIsInstance(clone["values"]["theta"], float)

    def test_int_binding_becomes_float(self):
        state = self.svc.create_parameter_state(rx_circuit(), {"theta": 1})
        self.assertEqual(state["values"], {"theta": 1.0})
        self.assertIsInstance(state["values"]["theta"], float)

    def test_inputs_not_modified(self):
        circuit = rx_circuit()
        values = {"theta": 0.3}
        circuit_snapshot = copy.deepcopy(circuit)
        values_snapshot = copy.deepcopy(values)
        self.svc.create_parameter_state(circuit, values)
        self.assertEqual(circuit, circuit_snapshot)
        self.assertEqual(values, values_snapshot)

    def test_circuit_validation_error_propagates(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.create_parameter_state({"qubit_count": -1}, {})

    def test_binding_error_propagates(self):
        with self.assertRaises(ParameterBindingError):
            self.svc.create_parameter_state(rx_circuit(), {})
        with self.assertRaises(ParameterBindingError):
            self.svc.create_parameter_state(rx_circuit(), {"theta": 0.1, "extra": 1.0})
        with self.assertRaises(ParameterBindingError):
            self.svc.create_parameter_state(rx_circuit(), {"theta": float("nan")})

    def test_package_exports_same_error(self):
        self.assertIs(PackageParameterServerError, ParameterServerError)


class ApplyParameterUpdatesTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.state = self.svc.create_parameter_state(
            two_param_circuit(), {"theta": 1.0, "phi": 2.0},
        )

    def apply(self, state=None, lr=0.1, staleness=2, updates=None):
        return self.svc.apply_parameter_updates(
            self.state if state is None else state,
            lr, staleness, updates,
        )

    def test_accepted_update_math_and_revision(self):
        result = self.apply(updates=[update("w1", 0, {"theta": 0.5, "phi": -1.0})])
        new_state = result["state"]
        self.assertEqual(new_state["revision"], 1)
        self.assertAlmostEqual(new_state["values"]["theta"], 1.0 - 0.1 * 0.5)
        self.assertAlmostEqual(new_state["values"]["phi"], 2.0 - 0.1 * (-1.0))
        self.assertEqual(result["results"], [
            {"id": "w1", "status": "accepted", "reason": None, "revision": 1},
        ])
        record = new_state["updates"][0]
        self.assertEqual(record["id"], "w1")
        self.assertEqual(record["base_revision"], 0)
        self.assertEqual(record["revision"], 1)
        self.assertEqual(
            record["digest"],
            digest_of("w1", 0, {"theta": 0.5, "phi": -1.0}),
        )

    def test_sequential_batches_accumulate(self):
        first = self.apply(updates=[update("w1", 0, {"theta": 1.0, "phi": 1.0})])
        second = self.apply(
            state=first["state"],
            updates=[update("w2", 1, {"theta": 1.0, "phi": 1.0})],
        )
        self.assertEqual(second["state"]["revision"], 2)
        self.assertAlmostEqual(second["state"]["values"]["theta"], 0.8)
        self.assertAlmostEqual(second["state"]["values"]["phi"], 1.8)
        self.assertEqual(len(second["state"]["updates"]), 2)

    def test_multiple_updates_in_one_call_apply_in_order(self):
        result = self.apply(updates=[
            update("w1", 0, {"theta": 1.0, "phi": 0.0}),
            update("w2", 1, {"theta": 1.0, "phi": 0.0}),
        ])
        self.assertEqual([r["status"] for r in result["results"]], ["accepted", "accepted"])
        self.assertEqual([r["revision"] for r in result["results"]], [1, 2])
        self.assertAlmostEqual(result["state"]["values"]["theta"], 0.8)

    def test_stale_update_rejected_without_state_change(self):
        base = self.apply(updates=[
            update("w1", 0, {"theta": 0.0, "phi": 0.0}),
            update("w2", 1, {"theta": 0.0, "phi": 0.0}),
            update("w3", 2, {"theta": 0.0, "phi": 0.0}),
        ])["state"]
        result = self.apply(state=base, staleness=1, updates=[
            update("w4", 1, {"theta": 5.0, "phi": 5.0}),
        ])
        self.assertEqual(result["results"], [
            {"id": "w4", "status": "rejected", "reason": "stale", "revision": 3},
        ])
        self.assertEqual(result["state"], base)

    def test_staleness_boundary_accepted(self):
        base = self.apply(updates=[
            update("w1", 0, {"theta": 0.0, "phi": 0.0}),
            update("w2", 1, {"theta": 0.0, "phi": 0.0}),
        ])["state"]
        result = self.apply(state=base, staleness=2, updates=[
            update("w3", 0, {"theta": 1.0, "phi": 1.0}),
        ])
        self.assertEqual(result["results"][0]["status"], "accepted")

    def test_future_revision_rejected_without_state_change(self):
        result = self.apply(updates=[update("w1", 3, {"theta": 1.0, "phi": 1.0})])
        self.assertEqual(result["results"], [
            {"id": "w1", "status": "rejected", "reason": "future_revision", "revision": 0},
        ])
        self.assertEqual(result["state"], self.state)

    def test_duplicate_returns_first_revision_without_reapply(self):
        first = self.apply(updates=[update("w1", 0, {"theta": 1.0, "phi": 1.0})])
        second = self.apply(
            state=first["state"],
            updates=[update("w1", 0, {"theta": 1.0, "phi": 1.0})],
        )
        self.assertEqual(second["results"], [
            {"id": "w1", "status": "duplicate", "reason": None, "revision": 1},
        ])
        self.assertEqual(second["state"], first["state"])

    def test_duplicate_within_single_call(self):
        result = self.apply(updates=[
            update("w1", 0, {"theta": 1.0, "phi": 1.0}),
            update("w1", 0, {"theta": 1.0, "phi": 1.0}),
        ])
        self.assertEqual(
            [r["status"] for r in result["results"]], ["accepted", "duplicate"],
        )
        self.assertEqual(result["state"]["revision"], 1)

    def test_same_id_different_content_fails_whole_call(self):
        first = self.apply(updates=[update("w1", 0, {"theta": 1.0, "phi": 1.0})])
        exc = ps_err(
            self.apply, state=first["state"],
            updates=[update("w1", 0, {"theta": 2.0, "phi": 1.0})],
        )
        self.assertEqual(exc.code, "idempotency_conflict")
        self.assertEqual(exc.path, "updates[0].id")

    def test_conflicting_ids_within_single_call_fail(self):
        exc = ps_err(self.apply, updates=[
            update("w1", 0, {"theta": 1.0, "phi": 1.0}),
            update("w1", 0, {"theta": 9.0, "phi": 1.0}),
        ])
        self.assertEqual(exc.code, "idempotency_conflict")
        self.assertEqual(exc.path, "updates[1].id")

    def test_json_round_trip_state_produces_identical_results(self):
        first = self.apply(updates=[update("w1", 0, {"theta": 1.0, "phi": 1.0})])
        clone = json.loads(json.dumps(first["state"]))
        updates = [
            update("w1", 0, {"theta": 1.0, "phi": 1.0}),
            update("w2", 1, {"theta": 0.5, "phi": 0.5}),
        ]
        from_original = self.apply(state=first["state"], updates=updates)
        from_clone = self.apply(state=clone, updates=updates)
        self.assertEqual(from_original, from_clone)

    def test_inputs_not_modified(self):
        state = self.svc.create_parameter_state(two_param_circuit(), {"theta": 1.0, "phi": 2.0})
        updates = [update("w1", 0, {"theta": 1.0, "phi": 1.0})]
        state_snapshot = copy.deepcopy(state)
        updates_snapshot = copy.deepcopy(updates)
        result = self.apply(state=state, updates=updates)
        self.assertEqual(state, state_snapshot)
        self.assertEqual(updates, updates_snapshot)
        self.assertIsNot(result["state"], state)

    def test_results_follow_update_order(self):
        result = self.apply(updates=[
            update("w1", 5, {"theta": 1.0, "phi": 1.0}),
            update("w2", 0, {"theta": 1.0, "phi": 1.0}),
            update("w3", 0, {"theta": 1.0, "phi": 1.0}),
        ])
        self.assertEqual([r["id"] for r in result["results"]], ["w1", "w2", "w3"])
        self.assertEqual(
            [r["status"] for r in result["results"]],
            ["rejected", "accepted", "accepted"],
        )


class ApplyParameterUpdatesValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.state = self.svc.create_parameter_state(rx_circuit(), {"theta": 0.5})
        self.updates = [update("w1", 0, {"theta": 1.0})]

    def apply(self, state=None, lr=0.1, staleness=2, updates=None):
        return self.svc.apply_parameter_updates(
            self.state if state is None else state,
            lr, staleness,
            self.updates if updates is None else updates,
        )

    def test_state_must_be_object(self):
        exc = ps_err(self.apply, state=[1, 2])
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state")

    def test_state_unknown_and_missing_fields(self):
        bad = dict(self.state, extra=1)
        exc = ps_err(self.apply, state=bad)
        self.assertEqual(exc.code, "invalid_state")
        exc = ps_err(self.apply, state={k: v for k, v in self.state.items() if k != "revision"})
        self.assertEqual(exc.code, "invalid_state")

    def test_state_version_must_be_one(self):
        exc = ps_err(self.apply, state=dict(self.state, version=2))
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state.version")

    def test_state_revision_must_be_non_negative_int(self):
        exc = ps_err(self.apply, state=dict(self.state, revision=-1))
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state.revision")
        exc = ps_err(self.apply, state=dict(self.state, revision=True))
        self.assertEqual(exc.code, "invalid_state")

    def test_state_values_must_cover_parameters_with_finite_numbers(self):
        exc = ps_err(self.apply, state=dict(self.state, values={}))
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state.values")
        exc = ps_err(self.apply, state=dict(self.state, values={"theta": 0.1, "x": 1}))
        self.assertEqual(exc.code, "invalid_state")
        exc = ps_err(self.apply, state=dict(self.state, values={"theta": float("inf")}))
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state.values.theta")

    def test_state_updates_length_must_match_revision(self):
        bad = dict(self.state, revision=1)
        exc = ps_err(self.apply, state=bad)
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state.updates")

    def test_broken_record_digest_rejected(self):
        accepted = self.apply()["state"]
        record = dict(accepted["updates"][0], digest="0" * 64)
        bad = dict(accepted, updates=[record])
        exc = ps_err(self.apply, state=bad)
        self.assertEqual(exc.code, "digest_mismatch")
        self.assertEqual(exc.path, "state.updates[0].digest")

    def test_record_revision_chain_enforced(self):
        accepted = self.apply()["state"]
        record = dict(accepted["updates"][0], revision=7)
        bad = dict(accepted, updates=[record])
        exc = ps_err(self.apply, state=bad)
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state.updates[0].revision")

    def test_learning_rate_validation(self):
        for bad in (0, -0.5, float("nan"), float("inf"), "0.1", True, None):
            exc = ps_err(self.apply, lr=bad)
            self.assertEqual(exc.code, "invalid_learning_rate")
            self.assertEqual(exc.path, "learning_rate")

    def test_max_staleness_validation(self):
        for bad in (-1, 0.5, "2", True, None):
            exc = ps_err(self.apply, staleness=bad)
            self.assertEqual(exc.code, "invalid_max_staleness")
            self.assertEqual(exc.path, "max_staleness")
        self.assertEqual(self.apply(staleness=0)["results"][0]["status"], "accepted")

    def test_updates_must_be_non_empty_array(self):
        for bad in ([], {}, "x", None):
            exc = ps_err(self.svc.apply_parameter_updates, self.state, 0.1, 2, bad)
            self.assertEqual(exc.code, "invalid_updates")
            self.assertEqual(exc.path, "updates")

    def test_update_structure_validation(self):
        exc = ps_err(self.apply, updates=[{"base_revision": 0, "gradients": {"theta": 1.0}}])
        self.assertEqual(exc.code, "invalid_update")
        self.assertEqual(exc.path, "updates[0]")
        exc = ps_err(self.apply, updates=[update("", 0, {"theta": 1.0})])
        self.assertEqual(exc.code, "invalid_update")
        self.assertEqual(exc.path, "updates[0].id")
        exc = ps_err(self.apply, updates=[update("w1", -1, {"theta": 1.0})])
        self.assertEqual(exc.code, "invalid_update")
        self.assertEqual(exc.path, "updates[0].base_revision")
        exc = ps_err(self.apply, updates=[update("w1", 0, {})])
        self.assertEqual(exc.code, "invalid_update")
        self.assertEqual(exc.path, "updates[0].gradients")
        exc = ps_err(self.apply, updates=[update("w1", 0, {"theta": 1.0, "x": 1.0})])
        self.assertEqual(exc.code, "invalid_update")
        exc = ps_err(self.apply, updates=[update("w1", 0, {"theta": float("nan")})])
        self.assertEqual(exc.code, "invalid_update")
        self.assertEqual(exc.path, "updates[0].gradients.theta")
        exc = ps_err(self.apply, updates=[dict(update("w1", 0, {"theta": 1.0}), extra=1)])
        self.assertEqual(exc.code, "invalid_update")

    def test_non_finite_computation_fails(self):
        huge = self.svc.create_parameter_state(rx_circuit(), {"theta": 1e308})
        exc = ps_err(
            self.apply, state=huge, lr=1e308,
            updates=[update("w1", 0, {"theta": -1e308})],
        )
        self.assertEqual(exc.code, "non_finite_result")
        self.assertEqual(exc.path, "updates[0].gradients.theta")

    def test_exception_returns_no_partial_state(self):
        try:
            self.apply(updates=[
                update("w1", 0, {"theta": 1.0}),
                update("w1", 0, {"theta": 2.0}),
            ])
        except ParameterServerError:
            pass
        else:
            raise AssertionError("ParameterServerError not raised")
        # 原状态未被修改，可继续正常使用。
        self.assertEqual(self.state["revision"], 0)
        result = self.apply()
        self.assertEqual(result["state"]["revision"], 1)


if __name__ == "__main__":
    unittest.main()
