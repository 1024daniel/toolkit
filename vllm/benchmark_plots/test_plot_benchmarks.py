#!/usr/bin/env python3
"""End-to-end regression tests; uses only Python's standard library."""

import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


SCRIPT = Path(__file__).with_name("plot_benchmarks.py")
EXPORTER = SCRIPT.parent.parent / "offhand_batch_test" / "export_data.py"
LEGACY_HEADERS = [
    "输入", "输出", "并发数", "单并发输出", "输出吞吐", "首token时延(ms)",
    "非首token时延(ms)", "测试时间", "测试时间段",
]
HEADERS = LEGACY_HEADERS + [
    "状态", "用例", "模型", "并行策略", "TP", "PP", "DP", "EP_ENABLED", "EP_SIZE",
]
SVG_NS = "{http://www.w3.org/2000/svg}"


def sample(**changes):
    row = dict(zip(HEADERS, [
        4096, 1024, 1, 100, 100, 30, 20, 60, "", "completed", "case",
        "model", "tp2", 2, 1, 1, 0, 1,
    ]))
    row.update(changes)
    return row


class PlotBenchmarksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="benchmark plots 测试 ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_rows(self, path, rows, *, legacy=False, encoding="utf-8-sig"):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding=encoding, newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LEGACY_HEADERS if legacy else HEADERS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def run_cli(self, *args, success=True, output="plots"):
        output = self.root / output
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args), "--output-dir", str(output)],
            capture_output=True, text=True, timeout=20,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        return result, output

    def read_csv(self, directory, filename):
        with (directory / filename).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def read_svg(self, directory, filename):
        tree = ET.parse(directory / filename)
        self.assertEqual(tree.getroot().tag, SVG_NS + "svg")
        # A well-formed XML document can still contain unusable SVG coordinates.
        for element in tree.iter():
            for attribute in ("x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "r",
                              "width", "height"):
                value = element.get(attribute)
                if value is not None and not value.endswith("%"):
                    self.assertTrue(math.isfinite(float(value)), (attribute, value))
        return tree

    def test_real_exporter_output_metadata_survives_rename_and_xml_characters(self):
        log = self.root / "group.log"
        log.write_text("""[bench] Model: 模型<&>
[bench] Strategy: 张量&专家 (TP=2 PP=1 DP=3 EP=true)
[bench] [1/1] Start benchmark: exported-case
[bench] Input: 4096 Output: 1024 Concurrency: 4 Prompts: 4
[bench] Benchmark duration (s): 60
[bench] Output token throughput (tok/s): 120
[bench] Mean TTFT (ms): 30
[bench] Mean TPOT (ms): 3.5
[bench] SUCCESS: exported-case; elapsed=60s
""", encoding="utf-8")
        # Managed columns must override all model/topology/workload hints in a name.
        source = self.root / "wrong_tp8pp9_in1_out1.csv"
        exported = subprocess.run([sys.executable, str(EXPORTER), str(log), "--output",
                                   str(source)], capture_output=True, text=True, timeout=20)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        self.assertTrue(source.read_bytes().startswith(b"\xef\xbb\xbf"))
        _, output = self.run_cli(source, "--metric", "all")
        rows = self.read_csv(output, "summary.csv")
        self.assertEqual(len(rows), 5)
        expected = {"tpot": 3.5, "ttft": 30, "throughput": 120,
                    "per-concurrency": 30, "duration": 60}
        for row in rows:
            self.assertEqual(row["model"], "模型<&>")
            self.assertEqual(row["strategy"], "张量&专家")
            self.assertEqual((row["tp"], row["pp"], row["dp"], row["ep_size"]),
                             ("2", "1", "3", "6"))
            self.assertEqual(float(row["value"]), expected[row["metric"]])
            self.assertEqual(row["n_valid"], "1")
            tree = self.read_svg(output, f'{row["metric"]}_in4096_out1024.svg')
            self.assertIn("模型<&>", "".join(tree.getroot().itertext()))
        original = self.read_csv(output, "samples.csv")
        self.assertEqual(len(original), 5)
        self.assertTrue(all(Path(row["source"]).name == source.name for row in original))
        self.assertTrue(all(row["row_number"] == "2" for row in original))

    def test_strategy_names_and_full_parallel_topology_keep_distinct_series(self):
        source = self.write_rows("renamed.csv", [
            sample(),
            sample(**{"DP": 2}),
            sample(**{"DP": 2, "EP_ENABLED": 1, "EP_SIZE": 4}),
            sample(**{"PP": 2, "DP": 2, "EP_ENABLED": 1, "EP_SIZE": 4}),
            sample(**{"并行策略": "different-kernel"}),
        ])
        _, output = self.run_cli(source)
        rows = self.read_csv(output, "summary.csv")
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({tuple(row[key] for key in
                                  ("strategy", "tp", "pp", "dp", "ep_enabled", "ep_size"))
                              for row in rows}), 5)
        _, filtered = self.run_cli(source, "--strategy", "different-kernel", output="filtered")
        self.assertEqual([row["strategy"] for row in self.read_csv(filtered, "summary.csv")],
                         ["different-kernel"])

    def test_workloads_split_and_explicit_filter_keeps_pairs(self):
        source = self.write_rows("matrix.csv", [
            sample(), sample(**{"输入": 512, "输出": 128, "非首token时延(ms)": 7}),
        ])
        _, output = self.run_cli(source)
        self.read_svg(output, "tpot_in4096_out1024.svg")
        self.read_svg(output, "tpot_in512_out128.svg")
        self.assertEqual(len(self.read_csv(output, "summary.csv")), 2)
        self.run_cli(source, "--input-len", "512", success=False, output="bad-pair")
        _, filtered = self.run_cli(source, "--input-len", "512", "--output-len", "128",
                                   output="filtered")
        rows = self.read_csv(filtered, "summary.csv")
        self.assertEqual([(row["input_len"], row["output_len"], float(row["value"]))
                          for row in rows], [("512", "128", 7.0)])
        self.assertFalse((filtered / "tpot_in4096_out1024.svg").exists())
        self.run_cli(source, "--input-len", "512", "--output-len", "1024",
                     success=False, output="absent")

    def test_three_datasets_ratios_exclude_failed_positive_and_missing_samples(self):
        baseline = self.write_rows("old/data.csv", [
            sample(), sample(**{"并发数": 2}), sample(**{"并发数": 3}),
            sample(**{"并发数": 5, "状态": "failed"}),
        ])
        candidate = self.write_rows("new/data.csv", [
            sample(**{"非首token时延(ms)": 10, "输出吞吐": 200}),
            sample(**{"并发数": 2, "状态": "failed", "非首token时延(ms)": 999}),
            sample(**{"并发数": 4}), sample(**{"并发数": 5}),
        ])
        nightly = self.write_rows("nightly/data.csv", [sample(**{"非首token时延(ms)": 15})])
        _, output = self.run_cli("--dataset", f"old={baseline}", "--dataset",
                                f"new&fast={candidate}", "--dataset", f"nightly={nightly}",
                                "--metric", "tpot", "--metric", "throughput")
        comparisons = self.read_csv(output, "comparison.csv")
        new_rows = {(row["metric"], int(row["concurrency"])): row
                    for row in comparisons if row["dataset"] == "new&fast"}
        self.assertAlmostEqual(float(new_rows["tpot", 1]["ratio"]), 0.5)
        self.assertAlmostEqual(float(new_rows["tpot", 1]["change_pct"]), -50)
        self.assertAlmostEqual(float(new_rows["tpot", 1]["delta"]), -10)
        self.assertAlmostEqual(float(new_rows["throughput", 1]["ratio"]), 2)
        self.assertAlmostEqual(float(new_rows["throughput", 1]["change_pct"]), 100)
        for concurrency in (2, 3, 4, 5):
            self.assertEqual(new_rows["tpot", concurrency]["ratio"], "")
        self.assertEqual(new_rows["tpot", 2]["n_valid"], "0")
        self.assertEqual(new_rows["tpot", 5]["baseline_n_valid"], "0")
        raw_failed = [row for row in self.read_csv(output, "samples.csv")
                      if row["dataset"] == "new&fast" and row["concurrency"] == "2"
                      and row["metric"] == "tpot"]
        self.assertEqual(float(raw_failed[0]["value"]), 999)
        self.assertIn("failed", raw_failed[0]["status"])
        nightly_row = next(row for row in comparisons if row["dataset"] == "nightly"
                           and row["metric"] == "tpot" and row["concurrency"] == "1")
        self.assertAlmostEqual(float(nightly_row["ratio"]), 0.75)
        for metric in ("tpot", "throughput"):
            self.read_svg(output, f"{metric}_in4096_out1024.svg")
            self.read_svg(output, f"{metric}_in4096_out1024_ratio.svg")

    def test_duplicate_runs_require_explicit_aggregation_and_ignore_failed_values(self):
        self.write_rows("runs/a.csv", [sample(**{"非首token时延(ms)": 10})])
        self.write_rows("runs/b.csv", [sample(**{"非首token时延(ms)": 20})])
        self.write_rows("runs/c.csv", [sample(**{"非首token时延(ms)": 30}),
                                        sample(**{"状态": "failed", "非首token时延(ms)": 900})])
        result, _ = self.run_cli(self.root / "runs", success=False)
        self.assertIn("duplicate", result.stderr.lower())
        for mode, expected in (("mean", 20), ("median", 20), ("min", 10), ("max", 30)):
            with self.subTest(aggregate=mode):
                _, output = self.run_cli(self.root / "runs", "--aggregate", mode, output=mode)
                rows = self.read_csv(output, "summary.csv")
                self.assertEqual(len(rows), 1)
                self.assertEqual(float(rows[0]["value"]), expected)
                self.assertEqual(rows[0]["n_total"], "4")
                self.assertEqual(rows[0]["n_valid"], "3")
                self.assertEqual(len(self.read_csv(output, "samples.csv")), 4)

    def test_strategy_chart_pairs_colors_and_separates_end_labels(self):
        source = self.write_rows("strategies.csv", [
            sample(**{"模型": "model", "并行策略": f"tp{tp}-ep{ep}", "TP": tp,
                      "PP": 8 // tp, "EP_ENABLED": ep, "EP_SIZE": tp if ep else 1,
                      "并发数": c, "非首token时延(ms)": 20})
            for tp in (1, 2, 4, 8) for ep in (0, 1) for c in (1, 2)
        ])
        _, output = self.run_cli(source)
        tree = self.read_svg(output, "tpot_in4096_out1024.svg")
        lines = [e for e in tree.findall(".//" + SVG_NS + "polyline")
                 if e.get("stroke-width") == "2.5"]
        self.assertEqual(len(lines), 8)
        colors = {e.get("stroke") for e in lines}
        self.assertEqual(len(colors), 4)
        for color in colors:
            self.assertEqual({e.get("stroke-dasharray") for e in lines if e.get("stroke") == color},
                             {"none", "8 5"})
        labels = [e for e in tree.findall(".//" + SVG_NS + "text")
                  if e.get("font-weight") == "600" and float(e.get("x")) > 1000]
        self.assertEqual(len(labels), 8)
        ys = sorted(float(e.get("y")) for e in labels)
        self.assertTrue(all(b - a >= 23 for a, b in zip(ys, ys[1:])))

    def test_environment_prefix_matches_and_exact_mode_preserves_names(self):
        baseline = self.write_rows("baseline.csv", [sample(模型="DeepSeek-V4-Flash")])
        candidate = self.write_rows("candidate.csv", [sample(模型="pcie-DeepSeek-V4-Flash")])
        args = ("--dataset", f"nvlink={baseline}", "--dataset", f"pcie={candidate}")
        _, output = self.run_cli(*args, "--model", "DeepSeek-V4-Flash")
        rows = self.read_csv(output, "comparison.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["model"], "DeepSeek-V4-Flash")
        _, output = self.run_cli(*args, "--match-model", "exact", output="exact")
        self.assertEqual([r["status"] for r in self.read_csv(output, "comparison.csv")],
                         ["missing", "missing"])

    def test_prefix_matching_does_not_join_different_models(self):
        baseline = self.write_rows("baseline.csv", [sample(模型="DeepSeek-V4-Flash")])
        candidate = self.write_rows("candidate.csv", [sample(模型="pcie-DeepSeek-V4-Pro")])
        _, output = self.run_cli("--dataset", f"a={baseline}", "--dataset", f"b={candidate}")
        self.assertTrue(all(r["status"] == "missing"
                            for r in self.read_csv(output, "comparison.csv")))

    def test_prefix_matching_rejects_ambiguity_and_collisions(self):
        for names, candidate_names, error in [
            (["a-model", "model"], ["pcie-a-model"], "ambiguous"),
            (["model"], ["model", "pcie-model"], "merge distinct"),
        ]:
            with self.subTest(names=names):
                baseline = self.write_rows("baseline.csv", [sample(模型=n) for n in names])
                candidate = self.write_rows("candidate.csv", [sample(模型=n) for n in candidate_names])
                result, _ = self.run_cli("--dataset", f"a={baseline}", "--dataset",
                                         f"b={candidate}", success=False)
                self.assertIn(error, result.stderr)

    def test_strategy_rename_needs_topology_matching(self):
        baseline = self.write_rows("old.csv", [sample(**{"并行策略": "before"})])
        candidate = self.write_rows("new.csv", [sample(**{"并行策略": "after",
                                                         "非首token时延(ms)": 10})])
        args = ("--dataset", f"old={baseline}", "--dataset", f"new={candidate}")
        _, separate = self.run_cli(*args)
        self.assertTrue(all(row["ratio"] == ""
                            for row in self.read_csv(separate, "comparison.csv")))
        _, matched = self.run_cli(*args, "--match-strategy", "topology", output="matched")
        rows = self.read_csv(matched, "comparison.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["ratio"]), 0.5)

    def test_topology_matching_never_joins_different_dp_or_ep(self):
        baseline = self.write_rows("old.csv", [sample(), sample(**{"并发数": 2, "DP": 2})])
        candidate = self.write_rows("new.csv", [sample(**{"DP": 2}),
                                                sample(**{"并发数": 2, "DP": 2,
                                                          "EP_ENABLED": 1, "EP_SIZE": 4})])
        _, output = self.run_cli("--dataset", f"old={baseline}", "--dataset",
                                f"new={candidate}", "--match-strategy", "topology")
        rows = self.read_csv(output, "comparison.csv")
        self.assertTrue(rows)
        self.assertTrue(all(row["ratio"] == "" for row in rows))

    def test_legacy_gb_encoding_and_explicit_prefix_removal(self):
        baseline = self.write_rows("Qwen_tp2pp1_in4096_out1024.csv", [sample()], legacy=True)
        candidate = self.write_rows("pcie_Qwen_tp2pp1_in4096_out1024.csv",
                                    [sample(**{"非首token时延(ms)": 10})], legacy=True,
                                    encoding="gb18030")
        _, output = self.run_cli("--dataset", f"old={baseline}", "--dataset",
                                f"pcie={candidate}", "--legacy-strip-prefix", "pcie=pcie_")
        rows = self.read_csv(output, "comparison.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows[0]["ratio"]), 0.5)
        self.assertEqual(rows[0]["tp"], "2")

    def test_invalid_dimensions_and_topology_fail_with_actionable_errors(self):
        for index, changes in enumerate(({"输入": "1.5"}, {"输出": -1}, {"并发数": 0},
                                          {"TP": 0}, {"DP": "NaN"},
                                          {"EP_ENABLED": 1, "EP_SIZE": 99})):
            with self.subTest(changes=changes):
                source = self.write_rows(f"invalid-{index}.csv", [sample(**changes)])
                result, _ = self.run_cli(source, success=False, output=f"bad-{index}")
                self.assertIn(source.name, result.stderr)

    def test_legacy_filename_workload_claim_is_checked_against_rows(self):
        source = self.write_rows("Qwen_tp2pp1_4k1k.csv",
                                 [sample(**{"输入": 512, "输出": 128})], legacy=True)
        result, _ = self.run_cli(source, success=False)
        self.assertIn("mismatch", result.stderr.lower())
        self.assertIn("4096/1024", result.stderr)
        self.assertIn("512/128", result.stderr)

    def test_managed_missing_identity_or_unknown_status_never_falls_back_to_filename(self):
        for index, changes in enumerate(({"模型": ""}, {"并行策略": ""},
                                          {"状态": ""}, {"状态": "success"})):
            with self.subTest(changes=changes):
                source = self.write_rows(f"model_tp2pp1_run{index}.csv", [sample(**changes)])
                result, _ = self.run_cli(source, success=False, output=f"rejected-{index}")
                self.assertIn(source.name, result.stderr)

    def test_nonfinite_negative_and_missing_metrics_are_retained_but_not_plotted(self):
        source = self.write_rows("metrics.csv", [
            sample(**{"并发数": index, "非首token时延(ms)": value})
            for index, value in enumerate(("nan", "inf", "-inf", -1, "", 42), 1)
        ])
        _, output = self.run_cli(source)
        rows = {row["concurrency"]: row for row in self.read_csv(output, "summary.csv")}
        self.assertEqual(len(rows), 6)
        for concurrency in range(1, 6):
            self.assertEqual(rows[str(concurrency)]["n_valid"], "0")
            self.assertEqual(rows[str(concurrency)]["value"], "")
        self.assertEqual(float(rows["6"]["value"]), 42)
        tree = self.read_svg(output, "tpot_in4096_out1024.svg")
        points = tree.findall(".//" + SVG_NS + 'g[@class="data-point"]')
        self.assertEqual(len(points), 1)

    def test_all_failed_metric_renders_without_invalid_axis_or_points(self):
        source = self.write_rows("failed.csv", [sample(**{"状态": "failed"}),
                                                 sample(**{"并发数": 2, "状态": "incomplete"})])
        _, output = self.run_cli(source)
        rows = self.read_csv(output, "summary.csv")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["n_valid"] == "0" and row["value"] == "" for row in rows))
        tree = self.read_svg(output, "tpot_in4096_out1024.svg")
        self.assertFalse(tree.findall(".//" + SVG_NS + 'g[@class="data-point"]'))

    def test_zero_policy_and_zero_baseline_do_not_create_infinite_ratios(self):
        baseline = self.write_rows("old.csv", [sample(**{"非首token时延(ms)": 0}),
                                               sample(**{"并发数": 2, "非首token时延(ms)": 5}),
                                               sample(**{"并发数": 3})])
        candidate = self.write_rows("new.csv", [sample(),
                                                sample(**{"并发数": 2, "非首token时延(ms)": 0}),
                                                sample(**{"并发数": 3, "状态": "failed",
                                                          "非首token时延(ms)": 0})])
        args = ("--dataset", f"old={baseline}", "--dataset", f"new={candidate}")
        _, excluded = self.run_cli(*args)
        self.assertTrue(all(row["ratio"] == ""
                            for row in self.read_csv(excluded, "comparison.csv")))
        _, included = self.run_cli(*args, "--include-zero", output="zeros")
        rows = {row["concurrency"]: row for row in self.read_csv(included, "comparison.csv")}
        self.assertEqual(float(rows["1"]["baseline_value"]), 0)
        self.assertEqual(rows["1"]["ratio"], "")
        self.assertEqual(float(rows["2"]["ratio"]), 0)
        self.assertEqual(float(rows["2"]["change_pct"]), -100)
        self.assertEqual(rows["3"]["ratio"], "")
        self.read_svg(included, "tpot_in4096_out1024.svg")
        self.read_svg(included, "tpot_in4096_out1024_ratio.svg")

    def test_recursive_input_ignores_its_generated_csvs_on_repeat_runs(self):
        self.write_rows("data/group-a/archive.csv", [sample()])
        self.write_rows("data/group-b/archive.csv", [sample(**{"并发数": 2})])
        args = (self.root / "data", "--glob", "*.csv")
        _, output = self.run_cli(*args, output="data/plots")
        first = self.read_csv(output, "summary.csv")
        self.assertEqual(len(first), 2)
        _, repeated = self.run_cli(*args, output="data/plots")
        self.assertEqual(self.read_csv(repeated, "summary.csv"), first)

    def test_output_never_overwrites_input_even_when_that_source_is_filtered_out(self):
        protected = self.write_rows("data/summary.csv", [sample(**{"模型": "other"})])
        self.write_rows("data/archive.csv", [sample()])
        original = protected.read_bytes()
        result, _ = self.run_cli(self.root / "data", "--model", "model", output="data",
                                 success=False)
        self.assertIn("overwrite", result.stderr.lower())
        self.assertEqual(protected.read_bytes(), original)

    def test_output_hardlink_to_input_is_rejected_without_changing_source(self):
        source = self.write_rows("data/archive.csv", [sample()])
        output = self.root / "plots"
        output.mkdir()
        alias = output / "samples.csv"
        os.link(source, alias)
        original = source.read_bytes()
        self.assertTrue(source.samefile(alias))
        result, _ = self.run_cli(source, output="plots", success=False)
        self.assertIn("overwrite", result.stderr.lower())
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(alias.read_bytes(), original)
        self.assertTrue(source.samefile(alias))

    def test_explicit_baseline_model_filter_and_plot_options(self):
        first = self.write_rows("first.csv", [sample(), sample(**{"模型": "other"})])
        second = self.write_rows("second.csv", [sample(**{"非首token时延(ms)": 10}),
                                                sample(**{"模型": "other"})])
        _, output = self.run_cli("--dataset", f"first={first}", "--dataset", f"second={second}",
                                "--baseline", "second", "--model", "model", "--linear-x",
                                "--y-scale", "independent", "--value-labels", "none",
                                "--columns", "1")
        rows = self.read_csv(output, "comparison.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["baseline"], "second")
        self.assertEqual(rows[0]["dataset"], "first")
        self.assertEqual(rows[0]["model"], "model")
        self.assertEqual(float(rows[0]["ratio"]), 2)
        self.read_svg(output, "tpot_in4096_out1024.svg")


if __name__ == "__main__":
    unittest.main()
