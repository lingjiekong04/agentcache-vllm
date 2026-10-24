"""Offline tool tests; synthetic tokenizer and HTTP, no vLLM/GPU import."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import demo
from workloads import SCENARIOS, generate


class Tools(unittest.TestCase):
    def test_http_timing_excludes_json_validation(self):
        events = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                events.append("closed")

            def read(self):
                events.append("read")
                return b'{"choices": [], "usage": {}}'

        def clock():
            events.append("clock")
            return 1 if len(events) == 1 else 1.025

        def validate(*args):
            events.append("validation")
            return [1]

        with (
            patch.object(demo.urllib.request, "urlopen", return_value=Response()),
            patch.object(demo.time, "perf_counter", side_effect=clock),
            patch.object(demo, "validate_response", side_effect=validate),
        ):
            row = demo.invoke({"request": {}})
        self.assertAlmostEqual(row["latency_ms"], 25)
        self.assertEqual(events, ["clock", "read", "closed", "clock", "validation"])

    def test_wrong_engine_or_source_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "server.log"
            with path.open("w") as out:
                out.write('AGENTCACHE_NATIVE_ENGINE {"config": {}}\n')
            with self.assertRaises(ValueError):
                demo.evidence(path, "quick", "two_queue", 1)

    def test_deterministic_inputs_counts_and_geometry(self):
        for profile in ("quick", "demo"):
            first = generate(profile, lambda s: [ord(c) for c in s])
            second = generate(profile, lambda s: [ord(c) for c in s])
            self.assertEqual(first, second)
            self.assertEqual(set(first["scenarios"]), set(SCENARIOS))
            self.assertEqual(
                [len(first["scenarios"][s]) for s in SCENARIOS], [17, 75, 30, 45]
            )
            for records in first["scenarios"].values():
                self.assertEqual(len({r["id"] for r in records}), len(records))

    def test_metric_missing_and_nonfinite_fail(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'vllm:gpu_cache_policy_cold_free_blocks{engine="0"} NaN\n'

        with patch.object(demo.urllib.request, "urlopen", return_value=Response()):
            with self.assertRaises(ValueError):
                demo.metrics()

    def test_invalid_response_and_logprobs_rejected(self):
        record = {"request": {"max_tokens": 1, "prompt": [1, 2]}}
        response = {
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "choices": [
                {
                    "finish_reason": "length",
                    "logprobs": {
                        "tokens": ["token_id:5"],
                        "token_logprobs": [0.0],
                        "top_logprobs": [{"token_id:5": 0.0}],
                    },
                }
            ],
        }
        self.assertEqual(demo.validate_response(record, response), [5])
        response["choices"][0]["logprobs"]["token_logprobs"] = [float("nan")]
        with self.assertRaises(ValueError):
            demo.validate_response(record, response)

    def test_failure_and_interrupt_saved_without_overwrite(self):
        for error in (ValueError("bad evidence"), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "result.json"
                args = SimpleNamespace(
                    output=path,
                    manifest=path,
                    server_log=path,
                    policy="lru",
                    scenario="scan_return",
                )
                with patch.object(demo, "read", side_effect=error):
                    with self.assertRaises(type(error)):
                        demo.run(args)
                self.assertFalse(json.loads(path.read_text())["complete"])
                with self.assertRaises(FileExistsError):
                    demo.run(args)

    def test_summary_rejects_invalid_timing(self):
        with self.assertRaises(ValueError):
            demo.summary(
                [
                    {
                        "phase": "cold",
                        "latency_ms": float("inf"),
                        "cache_queries": 1,
                        "cache_hits": 0,
                    }
                ]
            )

    def test_malformed_json_or_nan_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            demo.save(path, {"valid": 1})
            with self.assertRaises(FileExistsError):
                demo.save(path, {})
            with path.open("w") as out:
                out.write('{"a": NaN}')
            with self.assertRaises(ValueError):
                demo.read(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
