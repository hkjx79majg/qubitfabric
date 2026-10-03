import copy
import hashlib
import json
import math
import unittest

from qubitfabric import CacheStateError as PackageCacheStateError
from qubitfabric.service import (
    CacheStateError,
    CircuitValidationError,
    Service,
    SimulationError,
)


def cache_err(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except CacheStateError as exc:
        return exc
    raise AssertionError("CacheStateError not raised")


def digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedExpectationBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()
        self.circuit = {"qubit_count": 2, "operations": [
            {"gate": "h", "target": 0},
            {"gate": "cx", "control": 0, "target": 1},
        ]}
        self.observables = ["ZZ", "XX", "ZI"]

    def call(self, *args, **kwargs):
        return self.svc.cached_expectation(*args, **kwargs)

    def test_miss_then_hit_matches_expectation(self):
        first = self.call(self.circuit, self.observables)
        self.assertFalse(first["cache_hit"])
        self.assertEqual(
            first["result"],
            self.svc.expectation(self.circuit, self.observables),
        )
        request_id = first["request_id"]
        self.assertRegex(request_id, r"^[0-9a-f]{64}$")

        second = self.call(self.circuit, self.observables, cache=first["cache"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["request_id"], request_id)
        self.assertEqual(second["result"], first["result"])

    def test_request_id_is_sha256_of_bound_request(self):
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        response = self.call(circuit, ["Z"], values={"theta": math.pi})
        bound = self.svc.bind(circuit, {"theta": math.pi})
        payload = {
            "circuit": bound,
            "observables": ["Z"],
            "shots": None,
            "seed": None,
            "noise": {
                "single_qubit_depolarizing": 0.0,
                "two_qubit_depolarizing": 0.0,
            },
        }
        self.assertEqual(response["request_id"], digest(payload))

    def test_cache_shape_and_fields(self):
        response = self.call(self.circuit, self.observables)
        cache = response["cache"]
        self.assertEqual(set(cache), {"version", "entries"})
        self.assertEqual(cache["version"], 1)
        self.assertEqual(len(cache["entries"]), 1)
        entry = cache["entries"][0]
        self.assertEqual(set(entry), {"request_id", "result", "result_digest"})
        self.assertEqual(entry["request_id"], response["request_id"])
        self.assertEqual(entry["result"], response["result"])
        self.assertEqual(entry["result_digest"], digest(response["result"]))
        json.dumps(cache, sort_keys=True)

    def test_json_round_trip_keeps_identity_and_hit(self):
        first = self.call(self.circuit, self.observables)
        restored = json.loads(json.dumps(first["cache"]))
        second = self.call(self.circuit, self.observables, cache=restored)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["result"], first["result"])
        # 再来一次 JSON 往返仍然可用。
        third = self.call(
            self.circuit, self.observables,
            cache=json.loads(json.dumps(second["cache"])),
        )
        self.assertTrue(third["cache_hit"])

    def test_defaults_and_equivalent_number_forms_share_identity(self):
        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta", "coefficient": 1}},
        ]}
        variants = [
            self.call(circuit, ["Z"], values={"theta": 1}, noise={})["request_id"],
            self.call(circuit, ["Z"], values={"theta": 1.0})["request_id"],
            self.call(
                circuit, ["Z"],
                values={"theta": 1.0},
                noise={
                    "single_qubit_depolarizing": 0,
                    "two_qubit_depolarizing": 0.0,
                },
            )["request_id"],
            # 无参数电路：省略 values 与空映射等价。
            self.call({"qubit_count": 1}, ["Z"])["request_id"],
            self.call({"qubit_count": 1}, ["Z"], values={})["request_id"],
            # 采样省略 seed 采用默认 0。
            self.call({"qubit_count": 1}, ["Z"], shots=10)["request_id"],
            self.call({"qubit_count": 1}, ["Z"], shots=10, seed=0)["request_id"],
        ]
        self.assertEqual(len(set(variants[0:3])), 1)
        self.assertEqual(variants[3], variants[4])
        self.assertEqual(variants[5], variants[6])
        # 但精确请求与采样请求身份不同。
        self.assertNotEqual(variants[0], variants[3])
        self.assertNotEqual(variants[5], variants[3])

    def test_observable_order_and_duplicates_change_identity(self):
        base = self.call(self.circuit, ["ZZ", "XX"])["request_id"]
        swapped = self.call(self.circuit, ["XX", "ZZ"])["request_id"]
        duplicated = self.call(self.circuit, ["ZZ", "ZZ"])["request_id"]
        self.assertNotEqual(base, swapped)
        self.assertNotEqual(base, duplicated)
        # 重复项保留在结果中。
        response = self.call(self.circuit, ["ZZ", "ZZ"])
        self.assertEqual(
            [r["observable"] for r in response["result"]["results"]],
            ["ZZ", "ZZ"],
        )

    def test_binding_shots_seed_noise_change_identity(self):
        base = self.call(self.circuit, ["ZZ"], shots=100, seed=1)["request_id"]
        cases = [
            (self.circuit, ["ZZ"], {"shots": 101, "seed": 1}),
            (self.circuit, ["ZZ"], {"shots": 100, "seed": 2}),
            (self.circuit, ["ZZ"], {"shots": 100, "seed": 1,
                                    "noise": {"single_qubit_depolarizing": 0.01}}),
        ]
        ids = [self.call(c, obs, **kw)["request_id"] for c, obs, kw in cases]
        self.assertTrue(all(rid != base for rid in ids))
        self.assertEqual(len(set(ids)), 3)

        circuit = {"qubit_count": 1, "parameters": ["theta"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "theta"}},
        ]}
        a = self.call(circuit, ["Z"], values={"theta": 0.5})["request_id"]
        b = self.call(circuit, ["Z"], values={"theta": 0.6})["request_id"]
        self.assertNotEqual(a, b)

    def test_hit_moves_entry_to_front_and_eviction_pops_tail(self):
        a = self.call({"qubit_count": 1}, ["Z"], max_entries=2)
        b = self.call({"qubit_count": 1}, ["X"], cache=a["cache"], max_entries=2)
        self.assertEqual([e["request_id"] for e in b["cache"]["entries"]],
                         [b["request_id"], a["request_id"]])
        # 命中 a：a 提升到首位，b 沉到末位。
        hit_a = self.call(
            {"qubit_count": 1}, ["Z"], cache=b["cache"], max_entries=2,
        )
        self.assertTrue(hit_a["cache_hit"])
        self.assertEqual([e["request_id"] for e in hit_a["cache"]["entries"]],
                         [a["request_id"], b["request_id"]])
        # 新请求 c 插入首位，末项 b 被淘汰，a 保留。
        c = self.call(
            {"qubit_count": 1}, ["Y"], cache=hit_a["cache"], max_entries=2,
        )
        ids = [e["request_id"] for e in c["cache"]["entries"]]
        self.assertEqual(ids, [c["request_id"], a["request_id"]])
        # b 重新成为普通未命中，且原条目已不在快照中。
        b_again = self.call(
            {"qubit_count": 1}, ["X"], cache=c["cache"], max_entries=2,
        )
        self.assertFalse(b_again["cache_hit"])

    def test_capacity_one_keeps_only_latest(self):
        first = self.call({"qubit_count": 1}, ["Z"], max_entries=1)
        second = self.call(
            {"qubit_count": 1}, ["X"], cache=first["cache"], max_entries=1,
        )
        self.assertEqual(len(second["cache"]["entries"]), 1)
        self.assertEqual(
            second["cache"]["entries"][0]["request_id"], second["request_id"],
        )
        z_again = self.call(
            {"qubit_count": 1}, ["Z"], cache=second["cache"], max_entries=1,
        )
        self.assertFalse(z_again["cache_hit"])

    def test_default_capacity_is_128(self):
        cache = None
        first_id = None
        for shots in range(1, 129):
            response = self.call(
                {"qubit_count": 1}, ["Z"], shots=shots, seed=1, cache=cache,
            )
            if shots == 1:
                first_id = response["request_id"]
            cache = response["cache"]
        self.assertEqual(len(cache["entries"]), 128)
        # 默认容量始终生效：第 129 个请求插入后仍为 128，最久未用的末项淘汰。
        response = self.call(
            {"qubit_count": 1}, ["Z"], shots=129, seed=1, cache=cache,
        )
        self.assertEqual(len(response["cache"]["entries"]), 128)
        self.assertNotIn(
            first_id, [e["request_id"] for e in response["cache"]["entries"]],
        )

    def test_valid_snapshot_without_identity_is_plain_miss(self):
        first = self.call({"qubit_count": 1}, ["Z"])
        response = self.call(
            {"qubit_count": 1}, ["X"], cache=first["cache"],
        )
        self.assertFalse(response["cache_hit"])
        ids = [e["request_id"] for e in response["cache"]["entries"]]
        self.assertEqual(ids, [response["request_id"], first["request_id"]])

    def test_inputs_are_not_mutated(self):
        raw_circuit = {"qubit_count": 1, "parameters": ["t"], "operations": [
            {"gate": "rx", "target": 0, "angle": {"parameter": "t", "coefficient": 1}},
        ]}
        observables = ["Z", "X"]
        values = {"t": 0.5}
        snapshot_cache = {"version": 1, "entries": []}
        snapshots = (
            copy.deepcopy(raw_circuit),
            copy.deepcopy(observables),
            copy.deepcopy(values),
            copy.deepcopy(snapshot_cache),
        )
        self.call(
            raw_circuit, observables, values=values, shots=10, seed=1,
            cache=snapshot_cache, max_entries=3,
        )
        self.assertEqual(
            (raw_circuit, observables, values, snapshot_cache), snapshots,
        )

    def test_returned_result_and_cache_are_independent_copies(self):
        first = self.call(self.circuit, self.observables, shots=20, seed=1)
        first["result"]["results"][0]["counts"]["positive"] = -999
        second = self.call(self.circuit, self.observables, shots=20, seed=1,
                           cache=first["cache"])
        self.assertNotEqual(
            second["result"]["results"][0]["counts"]["positive"], -999,
        )
        # 返回的 cache 不与入参共享对象。
        source = first["cache"]
        response = self.call(self.circuit, self.observables, shots=20, seed=1,
                             cache=source)
        self.assertIsNot(response["cache"], source)
        response["cache"]["entries"].clear()
        self.assertTrue(source["entries"])
        # 命中结果与缓存内存储也不共享。
        response["result"]["results"][0]["observable"] = "TAMPERED"
        again = self.call(self.circuit, self.observables, shots=20, seed=1,
                          cache=response["cache"])
        self.assertNotEqual(
            again["result"]["results"][0]["observable"], "TAMPERED",
        )

    def test_sampling_without_seed_uses_default_seed(self):
        circuit = {"qubit_count": 1, "operations": [{"gate": "h", "target": 0}]}
        first = self.call(circuit, ["Z"], shots=50)
        second = self.call(circuit, ["Z"], shots=50, cache=first["cache"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["request_id"], second["request_id"])
        explicit = self.call(circuit, ["Z"], shots=50, seed=0, cache=second["cache"])
        self.assertTrue(explicit["cache_hit"])


class CachedExpectationValidationTest(unittest.TestCase):
    def setUp(self):
        self.svc = Service()

    def call(self, *args, **kwargs):
        return self.svc.cached_expectation(*args, **kwargs)

    def test_base_request_reuses_expectation_validation(self):
        with self.assertRaises(CircuitValidationError):
            self.call(
                {"qubit_count": 1, "operations": [{"gate": "x", "target": 5}]},
                ["Z"], max_entries=0,
            )
        with self.assertRaises(SimulationError) as ctx:
            self.call({"qubit_count": 1}, [], cache={"version": 1, "entries": []})
        self.assertEqual(
            (ctx.exception.code, ctx.exception.path),
            ("invalid_observables", "observables"),
        )
        with self.assertRaises(SimulationError) as ctx:
            self.call({"qubit_count": 21}, ["I" * 21])
        self.assertEqual(
            (ctx.exception.code, ctx.exception.path),
            ("state_space_too_large", "qubit_count"),
        )
        with self.assertRaises(SimulationError) as ctx:
            self.call({"qubit_count": 1}, ["Z"], seed=1)
        self.assertEqual(
            (ctx.exception.code, ctx.exception.path),
            ("seed_without_shots", "seed"),
        )

    def test_base_validation_precedes_cache_validation(self):
        # 基础请求错误优先于容量与快照错误。
        with self.assertRaises(CircuitValidationError):
            self.call(
                {"qubit_count": "nope"}, ["Z"],
                max_entries=True, cache="garbage",
            )
        # 容量校验先于快照校验。
        exc = cache_err(
            self.call, {"qubit_count": 1}, ["Z"],
            max_entries=0, cache="garbage",
        )
        self.assertEqual((exc.code, exc.path), ("invalid_cache_capacity", "max_entries"))

    def test_invalid_max_entries(self):
        for bad in (0, -1, 1.5, "10", True, False, [], None):
            with self.subTest(bad=bad):
                exc = cache_err(
                    self.call, {"qubit_count": 1}, ["Z"], max_entries=bad,
                )
                self.assertEqual(
                    (exc.code, exc.path),
                    ("invalid_cache_capacity", "max_entries"),
                )

    def test_invalid_cache_shapes(self):
        bad_caches = [
            "not-an-object",
            [],
            42,
            {"entries": []},
            {"version": 1},
            {"version": 1, "entries": [], "extra": 1},
            {"version": 0, "entries": []},
            {"version": "1", "entries": []},
            {"version": 1, "entries": {}},
        ]
        for bad in bad_caches:
            with self.subTest(bad=bad):
                exc = cache_err(self.call, {"qubit_count": 1}, ["Z"], cache=bad)
                self.assertEqual((exc.code, exc.path), ("invalid_cache", "cache"))

    def test_invalid_entries(self):
        result = self.svc.expectation({"qubit_count": 1}, ["Z"])
        rid = digest({"circuit": self.svc.bind({"qubit_count": 1}, {}),
                      "observables": ["Z"], "shots": None, "seed": None,
                      "noise": {"single_qubit_depolarizing": 0.0,
                                "two_qubit_depolarizing": 0.0}})
        good_entry = {"request_id": rid, "result": result, "result_digest": digest(result)}

        def with_entry(entry):
            return {"version": 1, "entries": [entry]}

        bad_entries = [
            with_entry("not-an-object"),
            with_entry({"request_id": rid, "result": result}),
            with_entry({"result": result, "result_digest": digest(result)}),
            with_entry({"request_id": rid, "result": result,
                        "result_digest": digest(result), "extra": 1}),
            with_entry({"request_id": "ABC" + "0" * 61, "result": result,
                        "result_digest": digest(result)}),
            with_entry({"request_id": "0" * 63, "result": result,
                        "result_digest": digest(result)}),
            with_entry({"request_id": 123, "result": result,
                        "result_digest": digest(result)}),
            with_entry({"request_id": rid, "result": result,
                        "result_digest": "0" * 63}),
            with_entry({"request_id": rid, "result": result,
                        "result_digest": "G" * 64}),
            with_entry({"request_id": rid, "result": result,
                        "result_digest": digest({"different": True})}),
            with_entry({"request_id": rid,
                        "result": {"qubit_count": 1, "shots": None,
                                   "results": [{"observable": "Z",
                                                "expectation": float("nan")}]},
                        "result_digest": "0" * 64}),
            with_entry({"request_id": rid, "result": {"native": {1, 2}},
                        "result_digest": "0" * 64}),
        ]
        for bad in bad_entries:
            with self.subTest(bad=bad):
                exc = cache_err(self.call, {"qubit_count": 1}, ["Z"], cache=bad)
                self.assertEqual((exc.code, exc.path), ("invalid_cache", "cache"))

        # 重复 request_id。
        duplicated = {"version": 1, "entries": [good_entry, copy.deepcopy(good_entry)]}
        exc = cache_err(self.call, {"qubit_count": 1}, ["Z"], cache=duplicated)
        self.assertEqual((exc.code, exc.path), ("invalid_cache", "cache"))

    def test_tampered_result_with_stale_digest_is_rejected(self):
        first = self.call({"qubit_count": 1}, ["Z"], shots=10, seed=1)
        tampered = copy.deepcopy(first["cache"])
        tampered["entries"][0]["result"]["results"][0]["observable"] = "X"
        # 摘要保持原值 → 载入即拒绝。
        exc = cache_err(
            self.call, {"qubit_count": 1}, ["Z"], shots=10, seed=1, cache=tampered,
        )
        self.assertEqual((exc.code, exc.path), ("invalid_cache", "cache"))

    def test_error_is_value_error_and_stable_export(self):
        self.assertIs(CacheStateError, PackageCacheStateError)
        exc = cache_err(
            self.call, {"qubit_count": 1}, ["Z"], max_entries=True,
        )
        self.assertIsInstance(exc, ValueError)
        self.assertEqual((exc.code, exc.path), ("invalid_cache_capacity", "max_entries"))
        exc = cache_err(
            self.call, {"qubit_count": 1}, ["Z"], cache={"version": 2, "entries": []},
        )
        self.assertEqual((exc.code, exc.path), ("invalid_cache", "cache"))


if __name__ == "__main__":
    unittest.main()
