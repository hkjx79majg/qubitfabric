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


def two_param_circuit():
    return {"qubit_count": 2, "parameters": ["theta", "phi"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        {"gate": "rz", "target": 1, "angle": {"parameter": "phi", "coefficient": 2.0}},
        {"gate": "cx", "control": 0, "target": 1},
    ]}


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(update_id, base_revision, gradients):
    payload = {"id": update_id, "base_revision": base_revision, "gradients": gradients}
    return hashlib.sha256(canon(payload).encode("utf-8")).hexdigest()


def update(update_id, base_revision, gradients):
    return {"id": update_id, "base_revision": base_revision, "gradients": gradients}


def server_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ParameterServerError as exc:
        return exc
    raise AssertionError("ParameterServerError not raised")


class CreateStateTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def test_state_shape_and_json_native(self):
        state = self.svc.create_parameter_state(two_param_circuit(), {"theta": 0.3, "phi": 1})
        self.assertEqual(set(state), {"version", "parameters", "revision", "values", "updates"})
        self.assertEqual(state["version"], 1)
        self.assertEqual(state["parameters"], ["theta", "phi"])
        self.assertEqual(state["revision"], 0)
        self.assertEqual(state["values"], {"theta": 0.3, "phi": 1.0})
        self.assertIsInstance(state["values"]["phi"], float)
        self.assertEqual(state["updates"], [])
        self.assertEqual(json.loads(json.dumps(state)), state)

    def test_negative_zero_canonicalized(self):
        state = self.svc.create_parameter_state(two_param_circuit(), {"theta": -0.0, "phi": 0.0})
        self.assertEqual(state["values"], {"theta": 0.0, "phi": 0.0})
        self.assertEqual(json.dumps(state["values"]), '{"theta": 0.0, "phi": 0.0}')

    def test_defaults_to_empty_values(self):
        state = self.svc.create_parameter_state({"qubit_count": 1})
        self.assertEqual(state["parameters"], [])
        self.assertEqual(state["values"], {})

    def test_normalization_semantics_reused(self):
        raw = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta", "coefficient": 0}},
        ]}
        state = self.svc.create_parameter_state(raw, {"theta": 2})
        self.assertEqual(state["parameters"], ["theta"])
        self.assertEqual(state["values"], {"theta": 2.0})

    def test_input_not_modified(self):
        circuit = two_param_circuit()
        values = {"theta": 0.3, "phi": 0.4}
        snapshot = copy.deepcopy((circuit, values))
        self.svc.create_parameter_state(circuit, values)
        self.assertEqual((circuit, values), snapshot)

    def test_circuit_validation_error_propagates(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.create_parameter_state({"qubit_count": -1}, {})

    def test_binding_errors_propagate(self):
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.create_parameter_state(two_param_circuit(), {"theta": 0.1})
        self.assertEqual(ctx.exception.code, "missing_parameter")
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.create_parameter_state(two_param_circuit(), {"theta": 0.1, "phi": 0.2, "x": 1})
        self.assertEqual(ctx.exception.code, "unknown_parameter")
        with self.assertRaises(ParameterBindingError) as ctx:
            self.svc.create_parameter_state(two_param_circuit(), {"theta": 0.1, "phi": float("nan")})
        self.assertEqual(ctx.exception.code, "non_finite_number")


class ApplyUpdatesTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.state = self.svc.create_parameter_state(two_param_circuit(), {"theta": 1.0, "phi": 2.0})

    def apply(self, state=None, updates=None, learning_rate=0.1, max_staleness=2):
        return self.svc.apply_parameter_updates(
            self.state if state is None else state,
            learning_rate,
            max_staleness,
            [update("u1", 0, {"theta": 0.5, "phi": -1.0})] if updates is None else updates,
        )

    def test_accept_single_update(self):
        result = self.apply()
        details = result["details"]
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0], {"id": "u1", "status": "accepted", "reason": None, "revision": 1})
        state = result["state"]
        self.assertEqual(state["revision"], 1)
        self.assertEqual(state["values"], {"theta": 1.0 - 0.1 * 0.5, "phi": 2.0 - 0.1 * (-1.0)})
        self.assertEqual(len(state["updates"]), 1)
        record = state["updates"][0]
        self.assertEqual(record["id"], "u1")
        self.assertEqual(record["revision"], 1)
        self.assertEqual(record["content"]["base_revision"], 0)
        self.assertEqual(record["content"]["gradients"], {"theta": 0.5, "phi": -1.0})
        self.assertEqual(record["digest"], digest_of("u1", 0, {"theta": 0.5, "phi": -1.0}))

    def test_sequential_updates_share_evolving_revision(self):
        result = self.apply(updates=[
            update("a", 0, {"theta": 1.0, "phi": 0.0}),
            update("b", 1, {"theta": 1.0, "phi": 0.0}),
            update("c", 0, {"theta": 1.0, "phi": 0.0}),
        ], max_staleness=1)
        self.assertEqual([d["status"] for d in result["details"]], ["accepted", "accepted", "rejected"])
        self.assertEqual(result["details"][2]["reason"], "stale")
        self.assertEqual(result["details"][2]["revision"], 2)
        self.assertEqual(result["state"]["revision"], 2)
        self.assertEqual(result["state"]["values"]["theta"], 1.0 - 0.1 - 0.1)

    def test_future_revision_rejected(self):
        result = self.apply(updates=[update("u1", 3, {"theta": 0.5, "phi": 0.0})])
        self.assertEqual(result["details"][0], {
            "id": "u1", "status": "rejected", "reason": "future_revision", "revision": 0,
        })
        self.assertEqual(result["state"], self.state)

    def test_stale_boundary(self):
        # max_staleness=0 时只有 base_revision 等于当前修订号才被接受。
        result = self.apply(updates=[
            update("a", 0, {"theta": 1.0, "phi": 0.0}),
            update("b", 0, {"theta": 1.0, "phi": 0.0}),
            update("c", 1, {"theta": 1.0, "phi": 0.0}),
        ], max_staleness=0)
        self.assertEqual(
            [(d["status"], d["reason"]) for d in result["details"]],
            [("accepted", None), ("rejected", "stale"), ("accepted", None)],
        )
        self.assertEqual(result["state"]["revision"], 2)

    def test_duplicate_returns_first_revision_without_reapplying(self):
        first = self.apply()
        replayed = self.apply(
            state=first["state"],
            updates=[update("u1", 0, {"theta": 0.5, "phi": -1.0})],
        )
        self.assertEqual(replayed["details"][0], {
            "id": "u1", "status": "duplicate", "reason": None, "revision": 1,
        })
        self.assertEqual(replayed["state"], first["state"])

    def test_duplicate_within_same_call(self):
        result = self.apply(updates=[
            update("a", 0, {"theta": 1.0, "phi": 0.0}),
            update("a", 0, {"theta": 1.0, "phi": 0.0}),
        ])
        self.assertEqual([d["status"] for d in result["details"]], ["accepted", "duplicate"])
        self.assertEqual(result["details"][1]["revision"], 1)
        self.assertEqual(result["state"]["revision"], 1)

    def test_duplicate_matches_canonical_content(self):
        # 数字等价形式（1 与 1.0）视为相同内容。
        first = self.apply(updates=[update("u1", 0, {"theta": 1, "phi": 0.0})])
        replayed = self.apply(
            state=first["state"],
            updates=[update("u1", 0, {"theta": 1.0, "phi": -0.0})],
        )
        self.assertEqual(replayed["details"][0]["status"], "duplicate")

    def test_idempotency_conflict_fails_whole_call(self):
        first = self.apply()
        exc = server_err(
            self.apply,
            state=first["state"],
            updates=[update("u1", 0, {"theta": 0.6, "phi": -1.0})],
        )
        self.assertEqual(exc.code, "idempotency_conflict")
        self.assertEqual(exc.path, "updates[0].id")

    def test_idempotency_conflict_within_same_call(self):
        exc = server_err(self.apply, updates=[
            update("a", 0, {"theta": 1.0, "phi": 0.0}),
            update("a", 0, {"theta": 2.0, "phi": 0.0}),
        ])
        self.assertEqual(exc.code, "idempotency_conflict")
        self.assertEqual(exc.path, "updates[1].id")

    def test_conflicting_base_revision_is_different_content(self):
        first = self.apply()
        exc = server_err(
            self.apply,
            state=first["state"],
            updates=[update("u1", 1, {"theta": 0.5, "phi": -1.0})],
        )
        self.assertEqual(exc.code, "idempotency_conflict")

    def test_non_finite_result_raises(self):
        exc = server_err(
            self.apply,
            updates=[update("u1", 0, {"theta": 1e308, "phi": 0.0})],
            learning_rate=1e308,
        )
        self.assertEqual(exc.code, "non_finite_result")
        self.assertEqual(exc.path, "updates[0].gradients.theta")

    def test_rejected_and_duplicate_do_not_change_revision(self):
        result = self.apply(updates=[
            update("a", 0, {"theta": 1.0, "phi": 0.0}),
            update("b", 9, {"theta": 1.0, "phi": 0.0}),
            update("a", 0, {"theta": 1.0, "phi": 0.0}),
        ])
        self.assertEqual(
            [(d["status"], d["revision"]) for d in result["details"]],
            [("accepted", 1), ("rejected", 1), ("duplicate", 1)],
        )
        self.assertEqual(result["state"]["revision"], 1)

    def test_state_survives_json_roundtrip(self):
        first = self.apply()
        snapshot = json.loads(json.dumps(first["state"]))
        direct = self.apply(state=first["state"], updates=[update("u2", 1, {"theta": 0.1, "phi": 0.2})])
        via_json = self.apply(state=snapshot, updates=[update("u2", 1, {"theta": 0.1, "phi": 0.2})])
        self.assertEqual(via_json, direct)
        json.dumps(via_json)

    def test_deterministic_replay(self):
        updates = [
            update("a", 0, {"theta": 0.3, "phi": 0.1}),
            update("b", 0, {"theta": 0.2, "phi": 0.4}),
            update("a", 0, {"theta": 0.3, "phi": 0.1}),
        ]
        self.assertEqual(self.apply(updates=updates), self.apply(updates=updates))

    def test_inputs_not_modified(self):
        updates = [update("a", 0, {"theta": 1.0, "phi": 0.0})]
        snapshot = copy.deepcopy((self.state, updates))
        result = self.apply(updates=updates)
        self.assertEqual((self.state, updates), snapshot)
        # 返回的状态是全新对象，不与输入共享结构。
        self.assertIsNot(result["state"], self.state)
        result["state"]["values"]["theta"] = 999
        result["state"]["updates"][0]["digest"] = "tampered"
        self.assertEqual(self.state["values"]["theta"], 1.0)
        self.assertEqual(self.state["updates"], [])

    def test_chained_calls_accumulate_records(self):
        first = self.apply(updates=[update("a", 0, {"theta": 1.0, "phi": 0.0})])
        second = self.apply(state=first["state"], updates=[update("b", 1, {"theta": 1.0, "phi": 0.0})])
        state = second["state"]
        self.assertEqual(state["revision"], 2)
        self.assertEqual([r["id"] for r in state["updates"]], ["a", "b"])
        self.assertEqual([r["revision"] for r in state["updates"]], [1, 2])
        self.assertAlmostEqual(state["values"]["theta"], 0.8)

    def test_empty_parameter_circuit(self):
        state = self.svc.create_parameter_state({"qubit_count": 1})
        result = self.svc.apply_parameter_updates(
            state, 0.5, 0, [update("u1", 0, {})],
        )
        self.assertEqual(result["details"][0]["status"], "accepted")
        self.assertEqual(result["state"]["revision"], 1)
        self.assertEqual(result["state"]["values"], {})


class ApplyValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.state = self.svc.create_parameter_state(two_param_circuit(), {"theta": 1.0, "phi": 2.0})
        self.updates = [update("u1", 0, {"theta": 0.5, "phi": 0.0})]

    def apply(self, state=None, learning_rate=0.1, max_staleness=2, updates=None):
        return self.svc.apply_parameter_updates(
            self.state if state is None else state,
            learning_rate,
            max_staleness,
            self.updates if updates is None else updates,
        )

    def broken_state(self, **changes):
        state = copy.deepcopy(self.state)
        state.update(changes)
        return state

    def test_state_must_be_object(self):
        exc = server_err(self.apply, state=[])
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state"))

    def test_state_unknown_and_missing_fields(self):
        exc = server_err(self.apply, state=self.broken_state(extra=1))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.extra"))
        state = self.broken_state()
        del state["revision"]
        exc = server_err(self.apply, state=state)
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.revision"))

    def test_state_version(self):
        exc = server_err(self.apply, state=self.broken_state(version=2))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.version"))
        exc = server_err(self.apply, state=self.broken_state(version=True))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.version"))

    def test_state_parameters(self):
        exc = server_err(self.apply, state=self.broken_state(parameters="theta"))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.parameters"))
        exc = server_err(self.apply, state=self.broken_state(parameters=["theta", "theta"]))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.parameters[1]"))

    def test_state_revision(self):
        for bad in (-1, 0.5, "0", None):
            exc = server_err(self.apply, state=self.broken_state(revision=bad))
            self.assertEqual((exc.code, exc.path), ("invalid_state", "state.revision"))

    def test_state_values(self):
        exc = server_err(self.apply, state=self.broken_state(values={"theta": 1.0}))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.values.phi"))
        exc = server_err(self.apply, state=self.broken_state(
            values={"theta": 1.0, "phi": 2.0, "gamma": 3.0}))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.values.gamma"))
        exc = server_err(self.apply, state=self.broken_state(
            values={"theta": float("inf"), "phi": 2.0}))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.values.theta"))
        exc = server_err(self.apply, state=self.broken_state(values={"theta": True, "phi": 2.0}))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.values.theta"))

    def test_state_record_structure(self):
        accepted = self.apply()
        record = accepted["state"]["updates"][0]

        state = self.broken_state(revision=1, updates=[{**record, "extra": 1}])
        exc = server_err(self.apply, state=state)
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.updates[0]"))

        state = self.broken_state(revision=1, updates=[{**record, "revision": 5}])
        exc = server_err(self.apply, state=state)
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.updates[0].revision"))

        state = self.broken_state(revision=1, updates=[record, record])
        exc = server_err(self.apply, state=state)
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.updates[1].revision"))

        other = copy.deepcopy(record)
        other["id"] = "u2"
        other["digest"] = digest_of("u2", 0, {"theta": 0.5, "phi": -1.0})
        state = self.broken_state(revision=1, updates=[record, other])
        exc = server_err(self.apply, state=state)
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.updates[1].revision"))

    def test_state_record_digest_mismatch(self):
        accepted = self.apply()
        record = accepted["state"]["updates"][0]

        tampered = {**record, "digest": "0" * 64}
        exc = server_err(self.apply, state=self.broken_state(revision=1, updates=[tampered]))
        self.assertEqual((exc.code, exc.path), ("digest_mismatch", "state.updates[0].digest"))

        tampered = copy.deepcopy(record)
        tampered["content"]["gradients"]["theta"] = 9.9
        exc = server_err(self.apply, state=self.broken_state(revision=1, updates=[tampered]))
        self.assertEqual((exc.code, exc.path), ("digest_mismatch", "state.updates[0].digest"))

        bad_format = {**record, "digest": "xyz"}
        exc = server_err(self.apply, state=self.broken_state(revision=1, updates=[bad_format]))
        self.assertEqual((exc.code, exc.path), ("invalid_state", "state.updates[0].digest"))

    def test_updates_array(self):
        exc = server_err(self.apply, updates=[])
        self.assertEqual((exc.code, exc.path), ("invalid_updates", "updates"))
        exc = server_err(self.apply, updates="u1")
        self.assertEqual((exc.code, exc.path), ("invalid_updates", "updates"))

    def test_update_structure(self):
        exc = server_err(self.apply, updates=[42])
        self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0]"))
        exc = server_err(self.apply, updates=[{**self.updates[0], "extra": 1}])
        self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].extra"))
        exc = server_err(self.apply, updates=[{"id": "u1", "base_revision": 0}])
        self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].gradients"))

    def test_update_id(self):
        for bad in ("", 7, None):
            exc = server_err(self.apply, updates=[update(bad, 0, {"theta": 0.5, "phi": 0.0})])
            self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].id"))

    def test_update_base_revision(self):
        for bad in (-1, 0.5, True, "0"):
            exc = server_err(self.apply, updates=[update("u1", bad, {"theta": 0.5, "phi": 0.0})])
            self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].base_revision"))

    def test_update_gradients(self):
        exc = server_err(self.apply, updates=[update("u1", 0, {"theta": 0.5})])
        self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].gradients.phi"))
        exc = server_err(self.apply, updates=[update("u1", 0, {"theta": 0.5, "phi": 0.0, "x": 1})])
        self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].gradients.x"))
        exc = server_err(self.apply, updates=[update("u1", 0, {"theta": 0.5, "phi": float("nan")})])
        self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].gradients.phi"))
        exc = server_err(self.apply, updates=[update("u1", 0, {"theta": 0.5, "phi": False})])
        self.assertEqual((exc.code, exc.path), ("invalid_update", "updates[0].gradients.phi"))

    def test_learning_rate(self):
        for bad in (0, -0.5, float("inf"), float("nan"), "0.1", True, None):
            exc = server_err(self.apply, learning_rate=bad)
            self.assertEqual((exc.code, exc.path), ("invalid_learning_rate", "learning_rate"))

    def test_max_staleness(self):
        for bad in (-1, 0.5, "2", True, None):
            exc = server_err(self.apply, max_staleness=bad)
            self.assertEqual((exc.code, exc.path), ("invalid_max_staleness", "max_staleness"))

    def test_validation_order_state_before_updates(self):
        exc = server_err(self.apply, state=self.broken_state(version=2), updates=[])
        self.assertEqual(exc.code, "invalid_state")
        exc = server_err(self.apply, updates=[], learning_rate=-1)
        self.assertEqual(exc.code, "invalid_updates")
        exc = server_err(self.apply, learning_rate=-1, max_staleness=-1)
        self.assertEqual(exc.code, "invalid_learning_rate")

    def test_exception_reports_first_error(self):
        updates = [
            update("ok", 0, {"theta": 0.5, "phi": 0.0}),
            update("bad", 0, {"theta": 0.5}),
            update("also-bad", -1, {"theta": 0.5, "phi": 0.0}),
        ]
        exc = server_err(self.apply, updates=updates)
        self.assertEqual(exc.path, "updates[1].gradients.phi")


class ExportTest(unittest.TestCase):
    def test_exception_exposed_from_package_and_service(self):
        self.assertIs(PackageParameterServerError, ParameterServerError)
        self.assertTrue(issubclass(ParameterServerError, ValueError))
        exc = ParameterServerError("invalid_state", "state")
        self.assertEqual(exc.code, "invalid_state")
        self.assertEqual(exc.path, "state")
        self.assertEqual(str(exc), "invalid_state at state")

    def test_existing_entries_unchanged(self):
        svc = Service()
        self.assertEqual(svc.health()["status"], "ok")
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        result = svc.expectation(circuit, ["Z"], values={"theta": 0.3})
        self.assertIn("results", result)
        gradient = svc.gradient(circuit, ["Z"], values={"theta": 0.3})
        self.assertIn("results", gradient)


if __name__ == "__main__":
    unittest.main()
