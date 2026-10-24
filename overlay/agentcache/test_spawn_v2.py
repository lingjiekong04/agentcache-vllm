"""Synthetic tests for the update, not a native Linux spawn acceptance."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_spawn_v2 as tool


class Tests(unittest.TestCase):
    def test_deadline_bounds(self):
        self.assertEqual(tool.timeout_value("600"), 600)
        for value in ("nan", "inf", "59", "901", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                tool.timeout_value(value)

    def test_native_result_contract(self):
        for policy, queue, capacity in (
            ("lru", "FreeKVCacheBlockQueue", 0),
            ("two_queue", "TwoQueuePolicy", 127),
        ):
            result = {
                "passed": True,
                "policy": policy,
                "module": str(tool.ROOT / "vllm/v1/core/block_pool.py"),
                "queue": queue,
                "stats": {
                    "protected_capacity": capacity,
                    "cold_free": 255,
                    "hot_free": 0,
                },
            }
            tool.validate_result(result, policy)
            result["queue"] = "UnrelatedQueue"
            with self.assertRaises(ValueError):
                tool.validate_result(result, policy)

    def test_progress_then_result(self):
        events = iter(
            [
                {"kind": "phase", "value": "native_imports"},
                {"kind": "result", "value": {"passed": True}},
            ]
        )
        peer = SimpleNamespace(poll=lambda _: True, recv=lambda: next(events))
        seen = []
        result = tool.wait_for_result(
            peer, None, 600, seen.append, clock=lambda: 0
        )
        self.assertTrue(result["passed"])
        self.assertEqual(len(seen), 2)

    def test_timeout_does_not_kill_child(self):
        clock = iter((0, 601))
        with self.assertRaises(TimeoutError):
            tool.wait_for_result(
                None, None, 600, lambda _: None, lambda: next(clock)
            )

    def test_exited_child_without_message(self):
        peer = SimpleNamespace(poll=lambda _: False)
        process = SimpleNamespace(is_alive=lambda: False)
        with self.assertRaises(RuntimeError):
            tool.wait_for_result(
                peer, process, 600, lambda _: None, clock=lambda: 0
            )

    def test_existing_output_rejected_before_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing.json"
            output.write_text("keep unchanged", encoding="utf-8")
            with (
                patch.object(
                    tool.sys,
                    "argv",
                    ["check", "--policy", "lru", "--output", str(output)],
                ),
                patch.object(tool.mp, "get_context") as context,
                self.assertRaises(FileExistsError),
            ):
                tool.main()
            context.assert_not_called()
            self.assertEqual(output.read_text(), "keep unchanged")

    def test_interruption_preserves_failed_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "interrupted.json"
            with (
                patch.object(
                    tool.sys,
                    "argv",
                    ["check", "--policy", "lru", "--output", str(output)],
                ),
                patch.dict(
                    tool.os.environ,
                    {"CUDA_VISIBLE_DEVICES": "0", "VLLM_USE_V1": "1"},
                ),
                patch.object(
                    tool.mp, "get_context", side_effect=KeyboardInterrupt
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as stopped,
            ):
                tool.main()
            self.assertEqual(stopped.exception.code, 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["error"], "KeyboardInterrupt()")
            self.assertFalse(report["child_still_running"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
