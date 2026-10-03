import copy
import hashlib
import json
import unittest

from qubitfabric import CacheStateError as PackageCacheStateError
from qubitfabric.service import (
    CacheStateError,
    CircuitValidationError,
    ParameterBindingError,
    Service,
    SimulationError,
)


def rx_circuit():
    return {"qubit_count": 1, "parameters": ["theta"], "operations": [
        {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
    ]}


def bell_circuit():
    return {"qubit_count": 2, "parameters": [], "operations": [
        {"gate": "h", "target": 0},
        {"gate": "cx", "control": 0, "target": 1},
    ]}


def empty_cache():
    return {"version": 1, "entries": []}


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_of(value):
    return hashlib.sha256(canon(value).encode("utf-8")).hexdigest()


def cache_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except CacheStateError as exc:
        return exc
    raise AssertionError("CacheStateError not raised")


class CachedExpectationBasicTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.observables = ["Z", "X"]
        self.values = {"theta": 0.3}

    def call(self, **overrides):
        kwargs = dict(
            circuit=self.circuit, observables=self.observables,
            values=self.values, cache=empty_cache(),
        )
        kwargs.update(overrides)
        circuit = kwargs.pop("circuit")
        observables = kwargs.pop("observables")
        values = kwargs.pop("values")
        return self.svc.cached_expectation(circuit, observables, values=values, **kwargs)

    def test_result_matches_direct_expectation_exact(self):
        response = self.call()
        direct = self.svc.expectation(
            self.circuit, self.observables, values=self.values,
        )
        self.assertEqual(response["result"], direct)
        self.assertFalse(response["cache_hit"])

    def test_result_matches_direct_expectation_sampled(self):
        response = self.call(shots=100, seed=7)
        direct = self.svc.expectation(
            self.circuit, self.observables, values=self.values, shots=100, seed=7,
        )
        self.assertEqual(response["result"], direct)

    def test_result_matches_direct_expectation_noisy(self):
        noise = {"single_qubit_depolarizing": 0.1, "two_qubit_depolarizing": 0.05}
        response = self.call(circuit=bell_circuit(), observables=["ZZ", "XX"],
                             values={}, noise=noise)
        direct = self.svc.expectation(
            bell_circuit(), ["ZZ", "XX"], values={}, noise=noise,
        )
        self.assertEqual(response["result"], direct)

    def test_request_id_format_and_stability(self):
        first = self.call()
        second = self.call()
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertIsInstance(first["request_id"], str)
        self.assertEqual(len(first["request_id"]), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in first["request_id"]))

    def test_second_call_hits_and_moves_entry_to_front(self):
        first = self.call()
        other = self.call(cache=first["cache"], shots=10)
        response = self.call(cache=other["cache"])
        self.assertTrue(response["cache_hit"])
        self.assertEqual(response["result"], first["result"])
        entries = response["cache"]["entries"]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["request_id"], response["request_id"])

    def test_cache_survives_json_roundtrip(self):
        first = self.call()
        snapshot = json.loads(json.dumps(first["cache"]))
        response = self.call(cache=snapshot)
        self.assertTrue(response["cache_hit"])
        json.dumps(response["cache"])

    def test_cache_structure(self):
        response = self.call()
        cache = response["cache"]
        self.assertEqual(set(cache), {"version", "entries"})
        self.assertEqual(cache["version"], 1)
        self.assertEqual(len(cache["entries"]), 1)
        entry = cache["entries"][0]
        self.assertEqual(set(entry), {"request_id", "result", "result_digest"})
        self.assertEqual(entry["request_id"], response["request_id"])
        self.assertEqual(entry["result"], response["result"])
        self.assertEqual(entry["result_digest"], digest_of(response["result"]))

    def test_inputs_not_modified(self):
        circuit = rx_circuit()
        observables = ["Z"]
        values = {"theta": 0.5}
        noise = {"single_qubit_depolarizing": 0.2}
        cache = empty_cache()
        snapshot = copy.deepcopy((circuit, observables, values, noise, cache))
        self.svc.cached_expectation(
            circuit, observables, values=values, shots=20, seed=3,
            noise=noise, cache=cache,
        )
        self.assertEqual((circuit, observables, values, noise, cache), snapshot)

    def test_hit_returns_independent_copies(self):
        first = self.call()
        response = self.call(cache=first["cache"])
        response["result"]["results"][0]["expectation"] = 999.0
        response["cache"]["entries"][0]["result"]["qubit_count"] = 99
        again = self.call(cache=first["cache"])
        self.assertEqual(again["result"], first["result"])
        self.assertEqual(again["cache"]["entries"][0]["result"], first["result"])

    def test_package_level_error_export(self):
        self.assertIs(PackageCacheStateError, CacheStateError)


class CachedExpectationIdentityTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.observables = ["Z", "X"]
        self.values = {"theta": 0.3}

    def request_id(self, **overrides):
        kwargs = dict(values=self.values, cache=empty_cache())
        kwargs.update(overrides)
        observables = kwargs.pop("observables", self.observables)
        circuit = kwargs.pop("circuit", self.circuit)
        values = kwargs.pop("values")
        return self.svc.cached_expectation(
            circuit, observables, values=values, **kwargs,
        )["request_id"]

    def test_default_omissions_do_not_change_identity(self):
        base = self.request_id()
        self.assertEqual(base, self.request_id(noise=None))
        self.assertEqual(base, self.request_id(noise={}))
        self.assertEqual(base, self.request_id(noise={
            "single_qubit_depolarizing": 0.0, "two_qubit_depolarizing": 0.0,
        }))
        free = {"qubit_count": 1, "operations": [{"gate": "x", "target": 0}]}
        self.assertEqual(
            self.request_id(circuit=free, values=None),
            self.request_id(circuit=free, values={}),
        )

    def test_default_seed_does_not_change_identity(self):
        self.assertEqual(
            self.request_id(shots=50),
            self.request_id(shots=50, seed=0),
        )

    def test_equivalent_numeric_forms_do_not_change_identity(self):
        base = self.request_id(values={"theta": 1})
        self.assertEqual(base, self.request_id(values={"theta": 1.0}))
        noisy = self.request_id(noise={"single_qubit_depolarizing": 1})
        self.assertEqual(noisy, self.request_id(noise={"single_qubit_depolarizing": 1.0}))
        circuit = {"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 1},
        ]}
        circuit_float = {"qubit_count": 1, "operations": [
            {"gate": "rx", "target": 0, "angle": 1.0},
        ]}
        self.assertEqual(
            self.request_id(circuit=circuit, values={}),
            self.request_id(circuit=circuit_float, values={}),
        )

    def test_observable_order_and_duplicates_matter(self):
        base = self.request_id()
        self.assertNotEqual(base, self.request_id(observables=["X", "Z"]))
        self.assertNotEqual(base, self.request_id(observables=["Z", "X", "Z"]))
        self.assertEqual(
            self.request_id(observables=["Z", "Z"]),
            self.request_id(observables=["Z", "Z"]),
        )

    def test_binding_shots_seed_noise_change_identity(self):
        base = self.request_id(shots=10, seed=1)
        self.assertNotEqual(base, self.request_id(shots=10, seed=1, values={"theta": 0.4}))
        self.assertNotEqual(base, self.request_id(shots=11, seed=1))
        self.assertNotEqual(base, self.request_id(shots=10, seed=2))
        self.assertNotEqual(base, self.request_id(
            shots=10, seed=1, noise={"single_qubit_depolarizing": 0.1},
        ))
        self.assertNotEqual(self.request_id(), self.request_id(shots=10))

    def test_unbound_circuit_changes_identity(self):
        other = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rz", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        self.assertNotEqual(self.request_id(), self.request_id(circuit=other))


class CachedExpectationEvictionTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.values = {"theta": 0.3}

    def test_lru_eviction(self):
        cache = empty_cache()
        ids = []
        for obs in (["Z"], ["X"], ["Y"]):
            response = self.svc.cached_expectation(
                self.circuit, obs, values=self.values, cache=cache, max_entries=2,
            )
            cache = response["cache"]
            ids.append(response["request_id"])
        self.assertEqual([e["request_id"] for e in cache["entries"]], [ids[2], ids[1]])

        # 命中 ["X"]：移到首位，["Z"] 已淘汰故再次未命中时淘汰 ["Y"]。
        hit = self.svc.cached_expectation(
            self.circuit, ["X"], values=self.values, cache=cache, max_entries=2,
        )
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(
            [e["request_id"] for e in hit["cache"]["entries"]], [ids[1], ids[2]],
        )
        miss = self.svc.cached_expectation(
            self.circuit, ["Z"], values=self.values, cache=hit["cache"], max_entries=2,
        )
        self.assertFalse(miss["cache_hit"])
        self.assertEqual(
            [e["request_id"] for e in miss["cache"]["entries"]], [ids[0], ids[1]],
        )

    def test_default_capacity_is_128(self):
        cache = empty_cache()
        for i in range(130):
            response = self.svc.cached_expectation(
                self.circuit, ["Z"], values={"theta": i + 0.5}, cache=cache,
            )
            cache = response["cache"]
        self.assertEqual(len(cache["entries"]), 128)

    def test_max_entries_one(self):
        cache = empty_cache()
        first = self.svc.cached_expectation(
            self.circuit, ["Z"], values=self.values, cache=cache, max_entries=1,
        )
        second = self.svc.cached_expectation(
            self.circuit, ["X"], values=self.values, cache=first["cache"], max_entries=1,
        )
        self.assertEqual(len(second["cache"]["entries"]), 1)
        self.assertEqual(second["cache"]["entries"][0]["request_id"], second["request_id"])


class CachedExpectationErrorTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.observables = ["Z"]
        self.values = {"theta": 0.3}

    def call(self, **overrides):
        kwargs = dict(values=self.values, cache=empty_cache())
        kwargs.update(overrides)
        return self.svc.cached_expectation(
            self.circuit, self.observables, **kwargs,
        )

    def test_invalid_max_entries(self):
        for bad in (0, -1, 1.5, True, "3", [], {}):
            with self.subTest(bad=bad):
                exc = cache_err(self.call, max_entries=bad)
                self.assertEqual(exc.code, "invalid_cache_capacity")
                self.assertEqual(exc.path, "max_entries")

    def test_invalid_cache_top_level(self):
        for bad in (None, [], "cache", 1, True):
            with self.subTest(bad=bad):
                exc = cache_err(self.call, cache=bad)
                self.assertEqual(exc.code, "invalid_cache")
                self.assertEqual(exc.path, "cache")

    def test_invalid_cache_version(self):
        for bad in (0, 2, "1", True, None):
            with self.subTest(bad=bad):
                exc = cache_err(self.call, cache={"version": bad, "entries": []})
                self.assertEqual(exc.code, "invalid_cache")
                self.assertEqual(exc.path, "cache")

    def test_invalid_cache_fields(self):
        exc = cache_err(self.call, cache={"entries": []})
        self.assertEqual(exc.code, "invalid_cache")
        exc = cache_err(self.call, cache={"version": 1})
        self.assertEqual(exc.code, "invalid_cache")
        exc = cache_err(self.call, cache={"version": 1, "entries": [], "extra": 1})
        self.assertEqual(exc.code, "invalid_cache")
        exc = cache_err(self.call, cache={"version": 1, "entries": {}})
        self.assertEqual(exc.code, "invalid_cache")

    def test_invalid_entry_structure(self):
        for bad_entry in (
            None, [], "x",
            {"request_id": "a" * 64, "result": {}},
            {"request_id": "a" * 64, "result": {}, "result_digest": "b" * 64, "x": 1},
        ):
            with self.subTest(bad_entry=bad_entry):
                exc = cache_err(self.call, cache={"version": 1, "entries": [bad_entry]})
                self.assertEqual(exc.code, "invalid_cache")
                self.assertEqual(exc.path, "cache")

    def test_invalid_request_id(self):
        result = {"qubit_count": 1, "shots": None, "results": []}
        digest = digest_of(result)
        for bad_id in ("a" * 63, "a" * 65, "A" * 64, "g" * 64, 1, None):
            with self.subTest(bad_id=bad_id):
                entry = {"request_id": bad_id, "result": result, "result_digest": digest}
                exc = cache_err(self.call, cache={"version": 1, "entries": [entry]})
                self.assertEqual(exc.code, "invalid_cache")

    def test_duplicate_request_id(self):
        result = {"qubit_count": 1, "shots": None, "results": []}
        entry = {
            "request_id": "a" * 64, "result": result,
            "result_digest": digest_of(result),
        }
        exc = cache_err(self.call, cache={"version": 1, "entries": [entry, dict(entry)]})
        self.assertEqual(exc.code, "invalid_cache")

    def test_invalid_digest(self):
        result = {"qubit_count": 1, "shots": None, "results": []}
        for bad_digest in ("b" * 63, "B" * 64, 1, None, digest_of({"other": 1})):
            with self.subTest(bad_digest=bad_digest):
                entry = {
                    "request_id": "a" * 64, "result": result,
                    "result_digest": bad_digest,
                }
                exc = cache_err(self.call, cache={"version": 1, "entries": [entry]})
                self.assertEqual(exc.code, "invalid_cache")

    def test_valid_snapshot_without_identity_is_plain_miss(self):
        result = {"qubit_count": 1, "shots": None, "results": []}
        entry = {
            "request_id": "a" * 64, "result": result,
            "result_digest": digest_of(result),
        }
        response = self.call(cache={"version": 1, "entries": [entry]})
        self.assertFalse(response["cache_hit"])
        self.assertEqual(len(response["cache"]["entries"]), 2)
        self.assertEqual(response["cache"]["entries"][1], entry)

    def test_base_validation_errors_unchanged(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.cached_expectation(
                {"qubit_count": -1}, self.observables, values={}, cache=empty_cache(),
            )
        with self.assertRaises(ParameterBindingError):
            self.svc.cached_expectation(
                self.circuit, self.observables, values={}, cache=empty_cache(),
            )
        with self.assertRaises(SimulationError) as ctx:
            self.svc.cached_expectation(
                self.circuit, ["Q"], values=self.values, cache=empty_cache(),
            )
        self.assertEqual(ctx.exception.code, "invalid_observable")
        with self.assertRaises(SimulationError) as ctx:
            self.svc.cached_expectation(
                self.circuit, self.observables, values=self.values,
                shots=0, cache=empty_cache(),
            )
        self.assertEqual(ctx.exception.code, "invalid_shots")
        with self.assertRaises(SimulationError) as ctx:
            self.svc.cached_expectation(
                self.circuit, self.observables, values=self.values,
                seed=1, cache=empty_cache(),
            )
        self.assertEqual(ctx.exception.code, "seed_without_shots")

    def test_qubit_limit_unchanged(self):
        circuit = {"qubit_count": 21, "operations": []}
        with self.assertRaises(SimulationError) as ctx:
            self.svc.cached_expectation(
                circuit, ["Z" * 21], values={}, cache=empty_cache(),
            )
        self.assertEqual(ctx.exception.code, "state_space_too_large")

    def test_base_validation_precedes_cache_validation(self):
        with self.assertRaises(CircuitValidationError):
            self.svc.cached_expectation(
                {"qubit_count": -1}, self.observables, values={},
                cache="not a cache", max_entries=0,
            )

    def test_max_entries_precedes_cache_validation(self):
        exc = cache_err(self.call, cache="not a cache", max_entries=0)
        self.assertEqual(exc.code, "invalid_cache_capacity")


class CachedExpectationSamplingTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = rx_circuit()
        self.values = {"theta": 0.3}

    def test_default_seed_matches_explicit_zero(self):
        implicit = self.svc.cached_expectation(
            self.circuit, ["Z"], values=self.values, shots=80, cache=empty_cache(),
        )
        explicit = self.svc.cached_expectation(
            self.circuit, ["Z"], values=self.values, shots=80, seed=0,
            cache=empty_cache(),
        )
        self.assertEqual(implicit["request_id"], explicit["request_id"])
        self.assertEqual(implicit["result"], explicit["result"])

    def test_sampled_hit_returns_same_counts(self):
        first = self.svc.cached_expectation(
            self.circuit, ["Z", "X"], values=self.values, shots=64, seed=11,
            cache=empty_cache(),
        )
        snapshot = json.loads(json.dumps(first["cache"]))
        second = self.svc.cached_expectation(
            self.circuit, ["Z", "X"], values=self.values, shots=64, seed=11,
            cache=snapshot,
        )
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["result"], first["result"])


if __name__ == "__main__":
    unittest.main()
