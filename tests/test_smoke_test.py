"""Regression tests for smoke-run error isolation."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test.py"
spec = importlib.util.spec_from_file_location("esl_smoke_test", MODULE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTestFailures(unittest.TestCase):
    def run_cases(self, side_effect):
        files = [smoke.REPO_ROOT / "first.py", smoke.REPO_ROOT / "second.py"]
        out = io.StringIO()
        with patch.object(smoke, "iter_py_files", return_value=files), patch.object(
            smoke.subprocess, "run", side_effect=side_effect
        ) as runner, contextlib.redirect_stdout(out):
            status = smoke.main()
        self.assertEqual(runner.call_count, 2)
        return status, out.getvalue()

    def test_timeout_does_not_abort_remaining_scripts(self):
        status, output = self.run_cases([
            subprocess.TimeoutExpired("python", 120),
            subprocess.CompletedProcess(["python"], 0, "", ""),
        ])
        self.assertEqual(status, 1)
        self.assertIn("Timed out after 120 seconds", output)
        self.assertIn("1/2 scripts passed.", output)

    def test_process_start_failure_does_not_abort(self):
        status, output = self.run_cases([
            OSError("executable missing"),
            subprocess.CompletedProcess(["python"], 0, "", ""),
        ])
        self.assertEqual(status, 1)
        self.assertIn("Could not start:", output)
        self.assertIn("1/2 scripts passed.", output)

    def test_nonzero_exit_is_reported(self):
        status, output = self.run_cases([
            subprocess.CompletedProcess(["python"], 2, "", "bad input"),
            subprocess.CompletedProcess(["python"], 0, "", ""),
        ])
        self.assertEqual(status, 1)
        self.assertIn("bad input", output)

    def test_all_success(self):
        ok = subprocess.CompletedProcess(["python"], 0, "", "")
        status, output = self.run_cases([ok, ok])
        self.assertEqual(status, 0)
        self.assertIn("2/2 scripts passed.", output)


if __name__ == "__main__":
    unittest.main()
