#!/usr/bin/env python3
"""Exercise orchestration with a fake vLLM process; no GPU or uv install needed.

Run with: python3 -m unittest discover -s scripts/offhand -p 'test_run_matrix.py' -v
The integration tests bind a local TCP port, just like the real vLLM service.
"""

import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_matrix.sh"

FAKE_UV = r'''#!/usr/bin/env python3
import http.server
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import urllib.request

args = sys.argv[1:]
args = args[args.index("vllm") + 1:]

def arg(name, default=None):
    try:
        return args[args.index(name) + 1]
    except ValueError:
        return default

def event(kind, **fields):
    fields.update(kind=kind, pid=os.getpid(), time=time.monotonic())
    with open(os.environ["FAKE_EVENTS"], "a", encoding="utf-8") as output:
        output.write(json.dumps(fields) + "\n")

def terminate(signum, frame):
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)

if args[0] == "serve":
    tp = arg("--tensor-parallel-size", "1")
    model = arg("--served-model-name", args[1])
    archived = sorted(str(path.relative_to(os.environ["FAKE_RUN_DIR"]))
                      for path in Path(os.environ["FAKE_RUN_DIR"]).glob("*/*.csv")
                      if path.is_file())
    event("start", tp=tp, model=model, model_path=args[1], archived_csvs=archived)
    if os.environ.get("FAKE_EXIT_TP") == tp:
        event("startup_exit", tp=tp)
        sys.exit(23)
    if os.environ.get("FAKE_WORKER") and os.fork() == 0:
        # Model workers can create a process group while keeping the server's
        # session. The supervisor must clean them up even if its leader exits.
        os.setpgid(0, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        event("worker", tp=tp)
        time.sleep(30)
        os._exit(0)
    if os.environ.get("FAKE_EXIT_DURING_CASE"):
        def die_after_case_begins():
            while True:
                try:
                    events = [json.loads(line) for line in
                              Path(os.environ["FAKE_EVENTS"]).read_text().splitlines()]
                    if any(item["kind"] == "case" for item in events):
                        event("service_exit", tp=tp)
                        os._exit(24)
                except (OSError, ValueError):
                    pass
                time.sleep(0.02)
        threading.Thread(target=die_after_case_begins, daemon=True).start()
    started = time.monotonic()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            ready = (not os.environ.get("FAKE_NEVER_READY")
                     and time.monotonic() - started >= 0.12)
            event("probe", tp=tp, path=self.path, ready=ready)
            payload = json.dumps({"data": [{"id": os.environ.get("FAKE_MODEL_ID", model)}]}).encode()
            self.send_response(200 if ready else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    class Server(http.server.HTTPServer):
        allow_reuse_address = True

    server = None
    try:
        server = Server((arg("--host", "127.0.0.1"), int(arg("--port", "8000"))), Handler)
        server.serve_forever(poll_interval=0.05)
    finally:
        if server is not None:
            server.server_close()
        event("stop", tp=tp, model=model)
else:
    assert args[:2] == ["bench", "serve"], args
    # A benchmark can only work after the service reports ready.
    with urllib.request.urlopen(arg("--base-url") + "/health", timeout=1) as response:
        assert response.status == 200
    event("case", input=arg("--random-input-len"), output=arg("--random-output-len"),
          concurrency=arg("--max-concurrency"), model=arg("--model"), tokenizer=arg("--tokenizer"))
    try:
        time.sleep(float(os.environ.get("FAKE_BENCH_SLEEP", "0")))
        requested = int(arg("--num-prompts"))
        completed = requested - 1 if os.environ.get("FAKE_PARTIAL") else requested
        result = Path(arg("--result-dir")) / arg("--result-filename")
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({"completed": completed, "requested": requested}))
        if (os.environ.get("FAKE_BLOCK_EXPORT") and
                "-single-" in result.name):
            # Force a real archive write failure without replacing the exporter.
            sys.path.insert(0, os.environ["FAKE_SCRIPT_DIR"])
            from export_data import archive_filename
            group = result.parent.parent
            topology = {name.lower(): int(os.environ[name]) for name in ("TP", "PP", "DP")}
            topology["ep"] = os.environ["EP"].lower() in ("1", "true", "yes")
            filename = archive_filename(
                os.environ["MODEL_NAME"], os.environ["STRATEGY_NAME"], topology,
                json.loads(os.environ["WORKLOADS_JSON"]),
                json.loads(os.environ["CONCURRENCIES_JSON"]),
                f"{group.parent.name}-g{group.name.split('-', 1)[0]}",
            )
            (group / filename).mkdir(exist_ok=True)
        print("FAKE_BENCH_DONE concurrency=" + arg("--max-concurrency"), flush=True)
        concurrency = int(arg("--max-concurrency"))
        print("============ Serving Benchmark Result ============")
        print("Successful requests:                     " + str(completed))
        print("Benchmark duration (s):                   0.50")
        print("Maximum request concurrency:              " + str(concurrency))
        print("Total input tokens:                       " +
              str(int(arg("--random-input-len")) * completed))
        print("Total generated tokens:                   " +
              str(int(arg("--random-output-len")) * completed))
        print("Request throughput (req/s):                " + str(completed * 2))
        print("Output token throughput (tok/s):           " + str(concurrency * 10))
        print("Total Token throughput (tok/s):            " + str(concurrency * 20))
        print("---------------Time to First Token----------------")
        print("Mean TTFT (ms):                           2.00")
        print("Median TTFT (ms):                         2.00")
        print("P99 TTFT (ms):                            2.00")
        print("-----Time per Output Token (excl. 1st token)-------")
        print("Mean TPOT (ms):                           3.00")
        print("Median TPOT (ms):                         3.00")
        print("P99 TPOT (ms):                            3.00")
        print("---------------Inter-token Latency----------------")
        print("Mean ITL (ms):                            3.00")
        print("Median ITL (ms):                          3.00")
        print("P99 ITL (ms):                             3.00", flush=True)
    finally:
        event("case_exit")
'''


class MatrixIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="offhand-matrix-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events_path = self.root / "events.jsonl"
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        fake_uv = fake_bin / "uv"
        fake_uv.write_text(textwrap.dedent(FAKE_UV), encoding="utf-8")
        fake_uv.chmod(0o755)
        self.env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))
        # Reserve an unused port for the short-lived service in this test.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.config = {
            "models": [{"name": "model-a", "path": "/fake/model-a"}],
            "strategies": [{"name": "single", "tp": 1, "pp": 1, "dp": 1},
                           {"name": "double", "tp": 2, "pp": 1, "dp": 1}],
            "workloads": [[16, 8]],
            "concurrencies": [1, 2],
            "env": {"FAKE_EVENTS": str(self.events_path),
                    "FAKE_RUN_DIR": str(self.root / "run"),
                    "FAKE_SCRIPT_DIR": str(SCRIPT_DIR)},
            "output_dir": str(self.root / "outputs"),
            "lock_file": str(self.root / "matrix.lock"),
            "server": {"host": "127.0.0.1", "port": port,
                       "startup_timeout": 3, "shutdown_timeout": 1,
                       "health_interval": 0.05, "health_timeout": 0.3,
                       "ready_successes": 2, "health_failure_threshold": 2,
                       "settle_seconds": 0},
            "benchmark": {"prompts_multiplier": 2, "case_timeout": 3,
                          "continue_on_error": False},
            "continue_on_error": False,
        }
        self.config_path = self.root / "matrix.json"
        self.run_dir = self.root / "run"
        self.addCleanup(self.cleanup_processes)

    def events(self):
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines() if line]

    @staticmethod
    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        # A zombie has already released its sockets and cannot execute code.
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"

    def cleanup_processes(self):
        if self.run_dir.exists() and RUNNER.exists():
            try:
                subprocess.run(["bash", str(RUNNER), "stop", "--run-dir", str(self.run_dir)],
                               env=self.env, capture_output=True, timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for event in self.events():
            if event["kind"] in {"start", "case", "worker"} and self.alive(event["pid"]):
                try:
                    os.kill(event["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def cli(self, command, *, timeout=15):
        self.write_config()
        args = ["bash", str(RUNNER), command]
        if command in {"run", "start", "dry-run"}:
            args.extend(["--config", str(self.config_path)])
        if command != "dry-run":
            args.extend(["--run-dir", str(self.run_dir)])
        return subprocess.run(args, cwd=self.root, env=self.env, text=True,
                              capture_output=True, timeout=timeout)

    def status(self):
        return json.loads((self.run_dir / "status.json").read_text())

    def group_statuses(self):
        return [json.loads(path.read_text())
                for path in sorted(self.run_dir.glob("*/status.json"))]

    def csv_rows(self, group):
        status = json.loads((group / "status.json").read_text())
        with Path(status["csv_file"]).open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    def assert_processes_stopped(self):
        for event in self.events():
            if event["kind"] in {"start", "case", "worker"}:
                self.assertFalse(self.alive(event["pid"]), event)

    def wait_for(self, predicate, timeout=6):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("Timed out waiting for expected runner state; events=" + repr(self.events()))

    def test_dry_run_has_no_process_side_effects(self):
        result = self.cli("dry-run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("model-a", result.stdout)
        self.assertIn("single", result.stdout)
        self.assertIn("double", result.stdout)
        self.assertEqual(self.events(), [])

    def test_enabled_ep_names_and_csv_use_tp_times_dp(self):
        self.config["strategies"] = [dict(name="expert", tp=2, pp=2, dp=2, ep=True)]
        result = self.cli("run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        group = next(self.run_dir.glob("001-*"))
        self.assertTrue(group.name.endswith("tp2-pp2-dp2-ep4"))
        for path in (group / "cases").glob("*.log"):
            self.assertIn("-tp2-pp2-dp2-ep4-", path.name)
        status = json.loads((group / "status.json").read_text())
        self.assertIn("-tp2-pp2-dp2-ep4__", Path(status["csv_file"]).name)
        for row in self.csv_rows(group):
            self.assertEqual(row["EP_ENABLED"], "1")
            self.assertEqual(row["EP_SIZE"], "4")

    def test_all_cases_share_server_and_groups_are_serial(self):
        self.config["models"].append({"name": "model-b", "path": "/fake/models second/main"})
        self.config["workloads"].append([32, 4])
        model_paths = {model["name"]: model["path"] for model in self.config["models"]}
        expected_cases = [(16, 8, 1), (16, 8, 2), (32, 4, 1), (32, 4, 2)]
        result = self.cli("run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "completed")
        events = self.events()
        starts = [e for e in events if e["kind"] == "start"]
        stops = [e for e in events if e["kind"] == "stop"]
        cases = [e for e in events if e["kind"] == "case"]
        expected_groups = [(name, tp) for name in model_paths for tp in ("1", "2")]
        self.assertEqual([(e["model"], e["tp"]) for e in starts], expected_groups)
        self.assertEqual([(e["model"], e["tp"]) for e in stops], expected_groups)
        self.assertEqual(len(cases), 16)
        # Archiving must finish before starting the next server configuration.
        self.assertEqual([len(e["archived_csvs"]) for e in starts], [0, 1, 2, 3])
        for stop, next_start in zip(stops, starts[1:]):
            self.assertLess(stop["time"], next_start["time"])
        for start, stop in zip(starts, stops):
            self.assertEqual(start["model_path"], model_paths[start["model"]])
            group_cases = [e for e in cases if start["time"] < e["time"] < stop["time"]]
            self.assertEqual([(int(e["input"]), int(e["output"]), int(e["concurrency"]))
                              for e in group_cases], expected_cases)
            for case in group_cases:
                self.assertEqual(case["model"], start["model"])
                self.assertEqual(case["tokenizer"], model_paths[start["model"]])
            ready = [e for e in events if e["kind"] == "probe" and e["ready"]
                     and e["path"] == "/v1/models"
                     and start["time"] < e["time"] < group_cases[0]["time"]]
            self.assertGreaterEqual(len(ready), 2)

        group_logs = sorted(self.run_dir.glob("*/group.log"))
        self.assertEqual(len(group_logs), 4)
        for group_log in group_logs:
            group = group_log.parent
            model_name = next(name for name in model_paths if name in group.name)
            strategy = next(item for item in self.config["strategies"]
                            if item["name"] in group.name)
            self.assertTrue((group / "server.log").is_file())
            self.assertEqual(group_log.read_text().count("FAKE_BENCH_DONE"), 4)
            case_logs = sorted((group / "cases").glob("*.log"))
            results = sorted((group / "results").glob("*.json"))
            self.assertEqual(len(case_logs), 4)
            self.assertEqual(len(results), 4)
            archive = self.csv_rows(group)
            self.assertEqual([(int(row["输入"]), int(row["输出"]), int(row["并发数"]))
                              for row in archive], expected_cases)
            for row in archive:
                self.assertEqual(float(row["单并发输出"]), 10)
                self.assertEqual(float(row["输出吞吐"]), int(row["并发数"]) * 10)
                self.assertEqual(float(row["首token时延(ms)"]), 2)
                self.assertEqual(float(row["非首token时延(ms)"]), 3)
                self.assertEqual(float(row["测试时间"]), 0.5)
                self.assertEqual(row["状态"], "completed")
                self.assertTrue(row["用例"].startswith(model_name))
                self.assertEqual(row["模型"], model_name)
                self.assertEqual(row["并行策略"], strategy["name"])
                self.assertEqual(int(row["TP"]), strategy["tp"])
                self.assertEqual(int(row["PP"]), strategy["pp"])
                self.assertEqual(int(row["DP"]), strategy["dp"])
                self.assertEqual(row["EP_ENABLED"], "0")
            group_status = json.loads((group / "status.json").read_text())
            self.assertEqual(group_status["export_state"], "completed")
            self.assertEqual(group_status["export_exit_code"], 0)
            csv_path = Path(group_status["csv_file"])
            self.assertEqual(csv_path.parent, group)
            self.assertEqual(csv_path.suffix, ".csv")
            for token in (model_name, strategy["name"],
                          f"tp{strategy['tp']}-pp1-dp1-epoff",
                          "in16-out8", "in32-out4", "c1-2",
                          f"{self.run_dir.name}-g{group.name.split('-', 1)[0]}"):
                self.assertIn(token, csv_path.name)
            self.assertLessEqual(len(csv_path.name.encode("ascii")), 240)
            self.assertIn("[export]", group_log.read_text())
            for log in case_logs:
                self.assertIn(model_name, log.name)
                self.assertIn("single" if "single" in group.name else "double", log.name)
                self.assertRegex(log.name, r"in16-out8|in32-out4")
                self.assertRegex(log.name, r"c[12](?:\D|$)")
                self.assertIn("FAKE_BENCH_DONE", log.read_text())
            for path in results:
                data = json.loads(path.read_text())
                self.assertEqual(data["completed"], data["requested"])
                self.assertIn(data["requested"], [2, 4])
        self.assertTrue((self.run_dir / "config.json").is_file())
        self.assertTrue((self.run_dir / "summary.json").is_file())
        self.assertEqual(json.loads((self.run_dir / "summary.json").read_text()),
                         self.group_statuses())
        self.assert_processes_stopped()

    def test_exit_zero_with_incomplete_requests_is_failed(self):
        self.config["strategies"] = self.config["strategies"][:1]
        self.config["env"]["FAKE_PARTIAL"] = "1"
        result = self.cli("run")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual(len([e for e in self.events() if e["kind"] == "case"]), 1)
        group = self.group_statuses()[0]
        self.assertEqual(group["export_state"], "completed")
        archive = self.csv_rows(Path(group["directory"]))
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0]["状态"], "failed")
        self.assertEqual(float(archive[0]["输出吞吐"]), 10)
        self.assert_processes_stopped()

    def test_startup_failure_can_continue_to_next_strategy(self):
        self.config["env"]["FAKE_EXIT_TP"] = "1"
        self.config["continue_on_error"] = True
        result = self.cli("run")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        events = self.events()
        self.assertEqual([e["tp"] for e in events if e["kind"] == "start"], ["1", "2"])
        self.assertEqual(len([e for e in events if e["kind"] == "case"]), 2)
        failed, completed = self.group_statuses()
        self.assertEqual(failed["export_state"], "skipped")
        self.assertEqual(list(Path(failed["directory"]).glob("*.csv")), [])
        self.assertEqual(completed["export_state"], "completed")
        self.assertEqual(len(self.csv_rows(Path(completed["directory"]))), 2)
        self.assert_processes_stopped()

    def test_export_failure_is_recorded_and_can_continue_after_cleanup(self):
        self.config["env"]["FAKE_BLOCK_EXPORT"] = "1"
        self.config["continue_on_error"] = True
        result = self.cli("run")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        failed, completed = self.group_statuses()
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["benchmark_exit_code"], 0)
        self.assertEqual(failed["export_state"], "failed")
        self.assertNotEqual(failed["export_exit_code"], 0)
        self.assertTrue(failed["export_error"])
        self.assertIn("export", failed["error"].lower())
        self.assertTrue(Path(failed["csv_file"]).is_dir())
        self.assertEqual(completed["state"], "completed")
        self.assertEqual(completed["export_state"], "completed")
        self.assertEqual(len(self.csv_rows(Path(completed["directory"]))), 2)
        starts = [e for e in self.events() if e["kind"] == "start"]
        stops = [e for e in self.events() if e["kind"] == "stop"]
        self.assertEqual([e["tp"] for e in starts], ["1", "2"])
        self.assertLess(stops[0]["time"], starts[1]["time"])
        self.assert_processes_stopped()

    def test_export_failure_stops_matrix_when_continue_on_error_is_false(self):
        self.config["env"]["FAKE_BLOCK_EXPORT"] = "1"
        result = self.cli("run")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual([e["tp"] for e in self.events() if e["kind"] == "start"], ["1"])
        groups = self.group_statuses()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["export_state"], "failed")
        self.assert_processes_stopped()

    def test_startup_timeout_stops_unready_service(self):
        self.config["strategies"] = self.config["strategies"][:1]
        self.config["server"]["startup_timeout"] = 0.5
        self.config["env"]["FAKE_NEVER_READY"] = "1"
        result = self.cli("run")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        self.assertTrue(any(e["kind"] == "start" for e in self.events()))
        self.assertFalse(any(e["kind"] == "case" for e in self.events()))
        self.assert_processes_stopped()

    def test_benchmark_timeout_stops_benchmark_and_service(self):
        self.config["strategies"] = self.config["strategies"][:1]
        self.config["benchmark"]["case_timeout"] = 1
        self.config["env"]["FAKE_BENCH_SLEEP"] = "30"
        result = self.cli("run", timeout=10)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual(len([e for e in self.events() if e["kind"] == "case"]), 1)
        self.assert_processes_stopped()

    def test_wrong_served_model_never_starts_benchmark(self):
        self.config["strategies"] = self.config["strategies"][:1]
        self.config["server"]["startup_timeout"] = 0.5
        self.config["env"]["FAKE_MODEL_ID"] = "unexpected-model"
        result = self.cli("run")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        events = self.events()
        self.assertTrue(any(e["kind"] == "probe" and e["path"] == "/v1/models" for e in events))
        self.assertFalse(any(e["kind"] == "case" for e in events))
        self.assert_processes_stopped()

    def test_occupied_port_leaves_foreign_service_running(self):
        port = self.config["server"]["port"]
        foreign = subprocess.Popen([sys.executable, "-m", "http.server", str(port),
                                    "--bind", "127.0.0.1"], cwd=self.root, env=self.env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def cleanup_foreign():
            if foreign.poll() is None:
                foreign.terminate()
                try:
                    foreign.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    foreign.kill()
                    foreign.wait(timeout=2)

        self.addCleanup(cleanup_foreign)

        def listening():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return True
            except OSError:
                return False

        self.wait_for(listening)
        result = self.cli("run")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual(self.events(), [], "An occupied port must block both server and benchmark launches")
        self.assertIsNone(foreign.poll(), "Runner terminated a service it does not own")
        self.assertTrue(listening(), "Foreign service no longer accepts connections")

    def test_service_death_stops_benchmark_and_separate_worker_group(self):
        self.config["strategies"] = self.config["strategies"][:1]
        self.config["benchmark"]["case_timeout"] = 60
        self.config["env"].update(FAKE_BENCH_SLEEP="30", FAKE_EXIT_DURING_CASE="1", FAKE_WORKER="1")
        result = self.cli("run", timeout=10)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.status()["state"], "failed")
        events = self.events()
        self.assertTrue(any(e["kind"] == "service_exit" for e in events))
        self.assertTrue(any(e["kind"] == "worker" for e in events))
        self.assertEqual(len([e for e in events if e["kind"] == "case"]), 1)
        self.assert_processes_stopped()

    def test_detached_start_status_and_stop(self):
        self.config["strategies"] = self.config["strategies"][:1]
        self.config["benchmark"]["case_timeout"] = 60
        self.config["env"]["FAKE_BENCH_SLEEP"] = "30"
        result = self.cli("start", timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(self.run_dir), result.stdout)
        self.wait_for(lambda: any(e["kind"] == "case" for e in self.events()))
        self.assertEqual(self.status()["state"], "running")
        runner_pid = self.status()["pid"]
        probes_before = sum(e["kind"] == "probe" for e in self.events())
        os.kill(runner_pid, signal.SIGHUP)
        # Continued model probes prove the supervisor still runs after hangup.
        self.wait_for(lambda: sum(e["kind"] == "probe" for e in self.events()) >= probes_before + 2)
        self.assertTrue(self.alive(runner_pid))
        self.assertEqual(self.status()["state"], "running")
        status_result = self.cli("status")
        self.assertEqual(status_result.returncode, 0, status_result.stdout + status_result.stderr)
        self.assertIn("running", status_result.stdout)
        stop_result = self.cli("stop")
        self.assertEqual(stop_result.returncode, 0, stop_result.stdout + stop_result.stderr)
        self.wait_for(lambda: self.status()["state"] == "stopped")
        self.assertTrue((self.run_dir / "runner.log").is_file())
        group = self.group_statuses()[0]
        self.assertEqual(group["state"], "stopped")
        self.assertEqual(group["export_state"], "completed")
        archive = self.csv_rows(Path(group["directory"]))
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0]["状态"], "incomplete")
        self.assertEqual(archive[0]["输入"], "16")
        self.assertEqual(archive[0]["输出吞吐"], "")
        self.assert_processes_stopped()


if __name__ == "__main__":
    unittest.main()
