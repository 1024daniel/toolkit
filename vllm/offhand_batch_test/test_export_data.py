#!/usr/bin/env python3
"""CSV regression tests; no GPU, service or third-party dependencies required."""

import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import export_data


SCRIPT = Path(__file__).with_name("export_data.py")
LEGACY = """INFO 09-15 12:00:00 importing vLLM
Namespace(model='a model, with punctuation', random_input_len=4096, random_output_len=1024, max_concurrency=4, num_prompts=4)
INFO 09-15 12:00:01 sending requests
Successful requests: 4
Benchmark duration (s): 2
Maximum request concurrency: 4
Output token throughput (tok/s): 120.0
Mean TTFT (ms): 12.5
Mean TPOT (ms): 3.5
Mean ITL (ms): 3.0
P99 ITL (ms): 4.0
"""


class ExportDataTests(unittest.TestCase):
    def test_legacy_columns_values_and_default_cli_destination(self):
        with tempfile.TemporaryDirectory(prefix="offhand export ") as directory:
            log = Path(directory) / "legacy benchmark.log"
            log.write_text(LEGACY, encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(log)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = log.with_suffix(".csv")
            self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))
            with output.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows, [export_data.HEADERS, [
                "4096", "1024", "4", "30.0", "120.0", "12.5", "3.5", "2",
                "09-15 12:00:00-09-15 12:00:03",
            ]])
            self.assertIn("Exported 1 cases", result.stdout)
            self.assertNotIn("4096,1024", result.stdout)

    def test_managed_prefixes_ansi_and_percentile_order(self):
        log = """[2026-09-15T12:00:00] Starting model
[server] [1/1] Start benchmark: misleading-server-header
[server] Output token throughput (tok/s): 9999
[bench] \x1b[32m[1/1] Start benchmark: model-tp8-in4096-out1024-c4\x1b[0m
[bench] Input: 4096  Output: 1024  Concurrency: 4  Prompts: 4
[bench] Namespace(random_input_len=4096, max_concurrency=None)
[server] Input: 9  Output: 9  Concurrency: 9  Prompts: 9
[bench] P99 ITL (ms): 42
[bench] Benchmark duration (s): 2
[bench] \x1b[1mOutput token throughput (tok/s):\x1b[0m 120.0
[server] Mean TPOT (ms): 9999
[bench] Mean TTFT (ms): 12.5
[bench] Mean TPOT (ms): 3.5
[bench] SUCCESS: model-tp8-in4096-out1024-c4; elapsed=3s
[export] Mean TPOT (ms): 9999
"""
        headers, rows = export_data.parse_log(io.StringIO(log))
        self.assertEqual(headers, export_data.HEADERS + export_data.MANAGED_HEADERS)
        self.assertEqual(rows, [[4096, 1024, 4, 30.0, 120.0, "12.5", "3.5", "2", "",
                                 "completed", "model-tp8-in4096-out1024-c4", *([None] * 7)]])

    def test_missing_percentile_failed_and_interrupted_cases_never_share_metrics(self):
        log = """[bench] [1/3] Start benchmark: completed-case
[bench] Input: 4096  Output: 1024  Concurrency: 4  Prompts: 4
[bench] Output token throughput (tok/s): 120
[bench] Mean TTFT (ms): 12.5
[bench] Mean TPOT (ms): 3.5
[bench] SUCCESS: completed-case; elapsed=3s
[bench] [2/3] Start benchmark: failed-case
[bench] Input: 512  Output: 128  Concurrency: 1  Prompts: 1
[bench] Mean TTFT (ms): 80
[bench] FAILURE: failed-case; command_exit=1; validation_exit=0; elapsed=1s
[bench] [3/3] Start benchmark: interrupted-case
[bench] Input: 1024  Output: 512  Concurrency: 2  Prompts: 2
[server] Output token throughput (tok/s): 9999
"""
        _, rows = export_data.parse_log(io.StringIO(log))
        self.assertEqual(len(rows), 3)
        self.assertEqual([row[10] for row in rows],
                         ["failed-case", "interrupted-case", "completed-case"])
        failed, interrupted, completed = rows
        self.assertEqual(failed[3:8], [None, None, "80", None, None])
        self.assertEqual(failed[9], "failed")
        self.assertEqual(interrupted[3:8], [None] * 5)
        self.assertEqual(interrupted[9], "incomplete")
        self.assertEqual(completed[3:7], [30.0, 120.0, "12.5", "3.5"])
        self.assertEqual(completed[9], "completed")

    def test_failed_last_case_without_namespace_or_metrics_is_archived(self):
        _, rows = export_data.parse_log(io.StringIO(
            "[1/1] Start benchmark: failed-before-launch\n"
            "Input: 512  Output: 32  Concurrency: 2  Prompts: 2\n"
            "FAILURE: unable to generate benchmark seed\n"
        ))
        self.assertEqual(rows, [[512, 32, 2, None, None, None, None, None, "",
                                 "failed", "failed-before-launch", *([None] * 7)]])

    def test_identity_inherits_group_header_and_preserves_unsanitized_names(self):
        log = '''[bench] Model: 原始 模型/a,b
[bench] Strategy: 默认 并行/策略
[bench] TP=8 PP=2 DP=1 EP=true
[server] Model: wrong-server-model
[export] Strategy: wrong-export-strategy
[bench] [1/2] Start benchmark: first
[bench] Input: 16  Output: 8  Concurrency: 1  Prompts: 1
[bench] Model: 精确 模型/"一",二
[bench] Strategy: 策略 一/二 (TP=4 PP=1 DP=2 EP=false)
[bench] Output token throughput (tok/s): 30
[server] Strategy: wrong (TP=100 PP=100 DP=100 EP=true)
[bench] SUCCESS: first; elapsed=1s
[bench] [2/2] Start benchmark: second
[bench] Input: 32  Output: 4  Concurrency: 2  Prompts: 2
[export] TP=200 PP=200 DP=200 EP=true
'''
        headers, rows = export_data.parse_log(io.StringIO(log))
        self.assertEqual(rows[0][11:], ['精确 模型/"一",二', "策略 一/二", 4, 1, 2, 0, 1])
        self.assertEqual(rows[1][11:], ["原始 模型/a,b", "默认 并行/策略", 8, 2, 1, 1, 8])
        self.assertEqual(rows[1][3:8], [None] * 5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "archive.csv"
            export_data.write_csv(output, headers, rows)
            with output.open(encoding="utf-8-sig", newline="") as stream:
                actual = list(csv.DictReader(stream))
            self.assertEqual(actual[0]["模型"], '精确 模型/"一",二')
            self.assertEqual(actual[0]["并行策略"], "策略 一/二")
            self.assertEqual(actual[1]["输出吞吐"], "")

    def test_ep_size_uses_tp_and_dp_not_pp(self):
        for tp, pp, dp, enabled, expected in [(4, 2, 2, True, 8),
                                             (1, 8, 1, True, 1),
                                             (8, 1, 2, False, 1)]:
            topology = dict(tp=tp, pp=pp, dp=dp, ep=enabled)
            log = (f"[1/1] Start benchmark: sample\n"
                   f"Strategy: test (TP={tp} PP={pp} DP={dp} EP={str(enabled).lower()})\n")
            headers, rows = export_data.parse_log(io.StringIO(log))
            result = dict(zip(headers, rows[0]))
            self.assertEqual(result["EP_ENABLED"], int(enabled))
            self.assertEqual(result["EP_SIZE"], expected)
            filename = export_data.archive_filename("model", "test", topology, [[16, 8]], [1], "run")
            self.assertIn(f"-ep{expected if enabled else 'off'}__", filename)

    def test_descriptive_filename_keeps_input_output_pairs_and_run_identity(self):
        filename = export_data.archive_filename(
            "DeepSeek-V4-Flash", "tp8", {"tp": 8, "pp": 1, "dp": 1, "ep": False},
            [(4096, 1024), (8192, 2048)], [1, 4, 16], "20260915-120000-123456-12345",
        )
        self.assertEqual(filename,
                         "DeepSeek-V4-Flash__tp8-tp8-pp1-dp1-epoff__"
                         "in4096-out1024+in8192-out2048__c1-4-16__20260915-120000-123456-12345.csv")
        self.assertNotIn("in4096-out2048", filename)

    def test_long_filename_is_bounded_distinct_safe_and_can_be_written(self):
        options = dict(
            model="模型 / Model " + "x" * 200, strategy="并行 / Strategy " + "y" * 200,
            topology={"tp": 2147483647, "pp": 2147483647, "dp": 2147483647, "ep": True},
            workloads=[(4096 + i, 1024 + i) for i in range(200)],
            concurrencies=list(range(1, 201)), run_tag="20260915-120000-123456-12345",
        )
        filename = export_data.archive_filename(**options)
        self.assertLessEqual(len(filename.encode("ascii")), 240)
        self.assertEqual(Path(filename).name, filename)
        self.assertNotIn(" ", filename)
        self.assertIn("Model", filename)
        self.assertIn("tp2147483647-pp2147483647-dp2147483647-ep4611686014132420609", filename)
        self.assertIn("in4096-out1024+more199io-", filename)
        self.assertIn("c1-200-n200-", filename)
        self.assertEqual(export_data.archive_filename(**options), filename)
        different = dict(options, workloads=[*options["workloads"][:-1], (9999, 9999)])
        self.assertNotEqual(export_data.archive_filename(**different), filename)
        unsafe_names = ["a/b", "a b", "a_b", "中文"]
        self.assertEqual(len({export_data.archive_filename(**dict(options, model=model))
                              for model in unsafe_names}), len(unsafe_names))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / filename
            export_data.write_csv(output, ["column"], [["value"]])
            self.assertTrue(output.is_file())
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_managed_cli_default_name_uses_actual_unique_matrix_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "group.log"
            log.write_text(
                "[bench] Model: model-a\n[bench] Strategy: pipeline\n"
                "[bench] TP=2 PP=4 DP=1 EP=0\n" + "".join(
                    f"[bench] [{index}/4] Start benchmark: case-{index}\n"
                    f"[bench] Input: {input_len}  Output: {output_len}  Concurrency: {concurrency}  Prompts: {concurrency}\n"
                    f"[bench] SUCCESS: case-{index}; elapsed=1s\n"
                    for index, (input_len, output_len, concurrency) in enumerate(
                        [(16, 8, 1), (16, 8, 2), (32, 4, 1), (32, 4, 2)], 1)), encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(log)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs = list(Path(directory).glob("*.csv"))
            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].name.startswith(
                "model-a__pipeline-tp2-pp4-dp1-epoff__in16-out8+in32-out4__c1-2__"))
            self.assertFalse(log.with_suffix(".csv").exists())

    def test_legacy_missing_values_and_multiple_blocks(self):
        _, rows = export_data.parse_log(io.StringIO(
            LEGACY + "Namespace(random_input_len=1, random_output_len=2, max_concurrency=None)\n"
            "Output token throughput (tok/s): nan\n"
            "Mean TTFT (ms): nan\n"
            "Mean TPOT (ms): N/A\n"
            "Benchmark duration (s): inf\n"
            "P99 ITL (ms): 4\n"
        ))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], [1, 2, None, None, None, None, None, None, ""])
        self.assertEqual(rows[1][:5], [4096, 1024, 4, 30.0, 120.0])

    def test_explicit_output_and_csv_escaping_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="offhand export ") as directory:
            log = Path(directory) / "group data.log"
            output = Path(directory) / "archived data.csv"
            log.write_text('[bench] [1/1] Start benchmark: case, "one"\n'
                           '[bench] Input: 1  Output: 2  Concurrency: 3  Prompts: 3\n'
                           '[bench] SUCCESS: case, "one"; elapsed=1s\n', encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(log), "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(log.with_suffix(".csv").exists())
            with output.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["用例"], 'case, "one"')
            self.assertEqual(rows[0]["状态"], "completed")
            self.assertEqual(rows[0]["输出吞吐"], "")

    def test_no_cases_cli_fails_without_publishing_or_overwriting_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "group.log"
            output = log.with_suffix(".csv")
            log.write_text("[server] Output token throughput (tok/s): 120\n"
                           "[bench] vLLM Benchmark Matrix\n", encoding="utf-8")
            for existing in (False, True):
                with self.subTest(existing_archive=existing):
                    if existing:
                        output.write_text("previous complete archive", encoding="utf-8")
                    result = subprocess.run([sys.executable, str(SCRIPT), str(log)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("no benchmark cases", result.stderr)
                    if existing:
                        self.assertEqual(output.read_text(), "previous complete archive")
                    else:
                        self.assertFalse(output.exists())

    def test_atomic_publish_failure_keeps_previous_archive_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "group.csv"
            output.write_text("previous complete archive", encoding="utf-8")
            with mock.patch.object(export_data.os, "replace", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    export_data.write_csv(output, ["column"], [["new data"]])
            self.assertEqual(output.read_text(), "previous complete archive")
            self.assertEqual(list(Path(directory).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
