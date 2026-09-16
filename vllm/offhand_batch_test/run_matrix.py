#!/usr/bin/env python3
"""Local Linux vLLM matrix supervisor; start.sh and batch.sh do the actual work."""

import argparse
import csv
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from export_data import archive_filename, topology_label


HERE = Path(__file__).resolve().parent
TERMINAL = {"completed", "failed", "stopped"}


def timestamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def slug(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")[:64] or "unnamed"


def positive(value, name, *, integer=False, zero=False, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or value < 0 or (not zero and value == 0)
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name}: expected {'nonnegative' if zero else 'positive'} "
                         f"{'integer' if integer else 'number'}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}: must be <= {maximum}")


def check_keys(obj, allowed, name):
    if not isinstance(obj, dict):
        raise ValueError(f"{name}: expected object")
    unknown = obj.keys() - set(allowed.split())
    if unknown:
        raise ValueError(f"{name}: unknown keys: {', '.join(sorted(unknown))}")


def check_options(obj, name):
    args = obj.setdefault("serve_args", [])
    if not isinstance(args, list) or any(not isinstance(x, str) or "\0" in x for x in args):
        raise ValueError(f"{name}.serve_args: expected array of strings")
    controlled = {"--model", "--host", "--port", "--served-model-name",
                  "--tensor-parallel-size", "--pipeline-parallel-size",
                  "--data-parallel-size", "--enable-expert-parallel",
                  "--no-enable-expert-parallel", "--config", "-tp", "-pp", "-dp", "-ep"}
    for arg in args:
        key = arg.split("=", 1)[0].replace("_", "-")
        # vLLM accepts underscore spellings and argparse abbreviations/short values.
        if (key in controlled or re.match(r"^-(tp|pp|dp)\d", key)
                or (key.startswith("--") and any(flag.startswith(key) for flag in controlled))):
            raise ValueError(f"{name}: configure {arg} through model/strategy/server fields")
    env = obj.setdefault("env", {})
    if not isinstance(env, dict) or any(
            not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
            or not isinstance(v, str) or "\0" in v for k, v in env.items()):
        raise ValueError(f"{name}.env: expected environment names and string values")


def load_config(path):
    c = json.loads(path.read_text())
    check_keys(c, "models strategies workloads concurrencies serve_args env output_dir "
               "work_dir lock_file server benchmark continue_on_error", "config")
    for key in ("models", "strategies", "workloads", "concurrencies"):
        if not isinstance(c.get(key), list) or not c[key]:
            raise ValueError(f"{key}: expected nonempty array")
    check_options(c, "config")
    for kind in ("models", "strategies"):
        names = set()
        for i, obj in enumerate(c[kind]):
            label = f"{kind}[{i}]"
            check_keys(obj, "name path serve_args env" if kind == "models" else
                       "name tp pp dp ep serve_args env", label)
            name = obj.get("name")
            if not isinstance(name, str) or not name.strip() or any(x in name for x in "\n\r\0"):
                raise ValueError(f"{label}.name: expected nonempty single-line string")
            if name in names:
                raise ValueError(f"{kind}: duplicate name {name!r}")
            names.add(name)
            check_options(obj, label)
            if kind == "models":
                if not isinstance(obj.get("path"), str) or not obj["path"].strip():
                    raise ValueError(f"{label}.path: expected model path or Hugging Face ID")
            else:
                for dim in ("tp", "pp", "dp"):
                    positive(obj.setdefault(dim, 1), f"{label}.{dim}", integer=True,
                             maximum=999999999)
                if not isinstance(obj.setdefault("ep", False), bool):
                    raise ValueError(f"{label}.ep: expected boolean")
    for pair in c["workloads"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("workloads: each item must be [input_len, output_len]")
        for value in pair:
            positive(value, "workload length", integer=True, maximum=2147483647)
    for value in c["concurrencies"]:
        positive(value, "concurrency", integer=True, maximum=2147483647)
    if len(set(map(tuple, c["workloads"]))) != len(c["workloads"]):
        raise ValueError("workloads: duplicate pair")
    if len(set(c["concurrencies"])) != len(c["concurrencies"]):
        raise ValueError("concurrencies: duplicate value")
    server = c.setdefault("server", {})
    defaults = dict(host="0.0.0.0", port=8000, startup_timeout=1800,
                    shutdown_timeout=120, health_interval=2, health_timeout=5,
                    ready_successes=2, health_failure_threshold=3, settle_seconds=5)
    check_keys(server, " ".join(defaults), "server")
    for key, value in defaults.items():
        server.setdefault(key, value)
    if not isinstance(server["host"], str) or not server["host"]:
        raise ValueError("server.host: expected local bind address")
    for key in ("port", "ready_successes", "health_failure_threshold"):
        positive(server[key], f"server.{key}", integer=True)
    if server["port"] > 65535:
        raise ValueError("server.port: must be <= 65535")
    for key in ("startup_timeout", "shutdown_timeout", "health_interval", "health_timeout",
                "settle_seconds"):
        positive(server[key], f"server.{key}", zero=(key == "settle_seconds"))
    bench = c.setdefault("benchmark", {})
    check_keys(bench, "request_rate prompts_multiplier case_timeout continue_on_error", "benchmark")
    bench.setdefault("request_rate", "inf")
    if isinstance(bench["request_rate"], bool) or not isinstance(bench["request_rate"], (str, int, float)):
        raise ValueError("benchmark.request_rate: expected inf or a positive rate")
    if str(bench["request_rate"]) != "inf":
        positive(float(bench["request_rate"]), "benchmark.request_rate")
    positive(bench.setdefault("prompts_multiplier", 1), "benchmark.prompts_multiplier",
             integer=True, maximum=2147483647)
    positive(bench.setdefault("case_timeout", 3600), "benchmark.case_timeout",
             integer=True, zero=True, maximum=2147483647)
    if max(c["concurrencies"]) * bench["prompts_multiplier"] > 2147483647:
        raise ValueError("concurrency * prompts_multiplier must be <= 2147483647")
    for obj in (c, bench):
        if not isinstance(obj.setdefault("continue_on_error", True), bool):
            raise ValueError("continue_on_error: expected boolean")
    for key, default in (("output_dir", "./offhand_runs"), ("work_dir", os.getcwd()),
                         ("lock_file", f"/tmp/offhand-vllm-{os.getuid()}.lock")):
        value = c.setdefault(key, default)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key}: expected nonempty path string")
        c[key] = str(Path(value).expanduser().resolve())
    if not Path(c["work_dir"]).is_dir():
        raise ValueError(f"work_dir does not exist: {c['work_dir']}")
    return c


def process_info(pid):
    try:
        # comm may contain spaces and parentheses; fields below start at state (3).
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"state": fields[0], "pgid": int(fields[2]),
                "sid": int(fields[3]), "ticks": fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def session_members(sid):
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            info = process_info(int(entry.name))
            if info and info["sid"] == sid and info["state"] not in ("Z", "X"):
                result.append((int(entry.name), info))
    return result


class Stopped(Exception):
    pass


class CleanupError(RuntimeError):
    pass


class Supervisor:
    def __init__(self, config, run_dir):
        self.c = config
        self.run_dir = run_dir
        self.stop_event = threading.Event()
        self.log_lock = threading.Lock()
        self.group_log = None
        self.children = []
        self.log_errors = []
        self.summary = []
        self.state = dict(state="running", pid=os.getpid(),
                          proc_start_ticks=process_info(os.getpid())["ticks"],
                          started_at=timestamp(), run_dir=str(run_dir))
        host = config["server"]["host"]
        self.connect_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        url_host = f"[{self.connect_host}]" if ":" in self.connect_host else self.connect_host
        self.base_url = f"http://{url_host}:{config['server']['port']}"
        # Local health checks must not be sent through HTTP_PROXY.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def save_state(self, **updates):
        self.state.update(updates, updated_at=timestamp())
        write_json(self.run_dir / "status.json", self.state)

    def log(self, text):
        line = f"[{timestamp()}] {text}\n"
        with self.log_lock:
            # A full log disk or disconnected foreground pipe must never skip cleanup.
            try:
                print(line, end="", flush=True)
                if self.group_log:
                    self.group_log.write(line)
                    self.group_log.flush()
            except (OSError, ValueError) as exc:
                self.log_errors.append(str(exc))

    def check_stop(self):
        if self.stop_event.is_set():
            raise Stopped("stop requested")
        if self.log_errors:
            raise RuntimeError(f"output logging failed: {self.log_errors[0]}")

    def pause(self, seconds):
        self.stop_event.wait(seconds)
        self.check_stop()

    def assert_port_free(self):
        server = self.c["server"]
        infos = socket.getaddrinfo(server["host"], server["port"], type=socket.SOCK_STREAM)
        for family, kind, proto, _, addr in infos:
            with socket.socket(family, kind, proto) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(addr)
                    sock.listen(1)
                except OSError as exc:
                    raise CleanupError(f"port {server['host']}:{server['port']} unavailable; "
                                       f"refusing to reuse/stop an existing service: {exc}") from exc

    def healthy(self, model_name):
        try:
            for endpoint in ("/health", "/v1/models"):
                with self.http.open(self.base_url + endpoint,
                                    timeout=self.c["server"]["health_timeout"]) as response:
                    if response.status != 200:
                        return False
                    if endpoint == "/v1/models":
                        data = json.load(response)
                        if model_name not in [m.get("id") for m in data.get("data", [])]:
                            return False
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, urllib.error.URLError):
            return False

    def launch(self, command, env, label, extra_log=None):
        self.log(f"{label} command: {shlex.join(command)}")
        proc = subprocess.Popen(command, cwd=self.c["work_dir"], env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                bufsize=1, start_new_session=True)
        # Register ownership before any subsequent operation can fail.
        child = dict(proc=proc, thread=None, label=label)
        self.children.append(child)

        def pump():
            try:
                with proc.stdout:
                    for line in proc.stdout:
                        with self.log_lock:
                            if extra_log:
                                extra_log.write(line)
                                extra_log.flush()
                            self.group_log.write(f"[{label}] {line}")
                            self.group_log.flush()
            except (OSError, ValueError) as exc:
                self.log_errors.append(str(exc))

        child["thread"] = threading.Thread(target=pump, daemon=True)
        child["thread"].start()
        return proc

    def cleanup(self):
        # TERM/KILL every process in sessions we created, including DP worker groups.
        # Ignore zombies: they have released sockets/GPU resources and cannot be killed.
        errors = []
        for child in reversed(self.children):
            proc = child["proc"]
            proc.poll()

            def send(sig):
                for pid, info in session_members(proc.pid):
                    current = process_info(pid)
                    if current and current["ticks"] == info["ticks"] and current["sid"] == proc.pid:
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass

            if session_members(proc.pid):
                self.log(f"Stopping {child['label']} session {proc.pid}")
                send(signal.SIGTERM)
                deadline = time.monotonic() + self.c["server"]["shutdown_timeout"]
                while session_members(proc.pid) and time.monotonic() < deadline:
                    proc.poll()
                    time.sleep(0.1)
                if session_members(proc.pid):
                    self.log(f"Graceful shutdown timed out; KILL {child['label']} session {proc.pid}")
                    send(signal.SIGKILL)
                    deadline = time.monotonic() + 5
                    while session_members(proc.pid) and time.monotonic() < deadline:
                        proc.poll()
                        time.sleep(0.1)
                if session_members(proc.pid):
                    errors.append(f"session {proc.pid} still has live processes")
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                errors.append(f"cannot reap {child['label']} PID {proc.pid}")
            if child["thread"]:
                child["thread"].join(timeout=2)
                if child["thread"].is_alive():
                    errors.append(f"{child['label']} output stream still open")
        self.children.clear()
        if errors:
            raise CleanupError("; ".join(errors))

    def wait_ready(self, proc, model):
        s = self.c["server"]
        deadline = time.monotonic() + s["startup_timeout"]
        consecutive = 0
        while time.monotonic() < deadline:
            self.check_stop()
            if proc.poll() is not None:
                raise RuntimeError(f"server exited before readiness: exit={proc.returncode}")
            ready = self.healthy(model)
            if proc.poll() is not None:
                raise RuntimeError(f"server exited during readiness: exit={proc.returncode}")
            consecutive = consecutive + 1 if ready else 0
            if consecutive >= s["ready_successes"]:
                self.log(f"Server ready: {self.base_url}, model={model}")
                return
            self.pause(min(s["health_interval"], max(0, deadline - time.monotonic())))
        raise RuntimeError(f"server startup timed out after {s['startup_timeout']}s")

    def export_group(self, folder, result, model, strategy):
        case_count = len(list((folder / "cases").glob("*.log")))
        if not case_count:
            result["export_state"] = "skipped"
            self.log("CSV export skipped: no benchmark cases were recorded")
            return

        # Include the group index as well as the run identifier so CSVs remain
        # distinct when copied out of their directories for delivery.
        run_tag = f"{self.run_dir.name}-g{folder.name.split('-', 1)[0]}"
        csv_file = folder / archive_filename(
            model["name"], strategy["name"], strategy,
            self.c["workloads"], self.c["concurrencies"], run_tag,
        )
        result.update(csv_file=str(csv_file), export_state="failed")
        self.save_state(phase="exporting")
        self.log(f"Archiving {case_count} cases to {csv_file}")
        command = [sys.executable, str(HERE / "export_data.py"),
                   str(folder / "group.log"), "--output", str(csv_file)]
        try:
            proc = self.launch(command, os.environ | {"PYTHONUNBUFFERED": "1"}, "export")
            try:
                result["export_exit_code"] = proc.wait(timeout=300)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("CSV export timed out after 300s") from exc
        finally:
            # Services and benchmarks have already been cleaned up. Only the
            # exporter remains here; drain its output before closing group.log.
            self.cleanup()

        if result["export_exit_code"] != 0:
            raise RuntimeError(f"export_data.py failed: exit={result['export_exit_code']}; see group.log")
        with csv_file.open(newline="", encoding="utf-8-sig") as stream:
            rows = sum(1 for _ in csv.DictReader(stream))
        if rows != case_count:
            raise RuntimeError(f"CSV contains {rows} cases; expected {case_count} recorded case logs")
        result.update(export_state="completed", csv_rows=rows)
        self.log(f"CSV archive completed: {csv_file} ({rows} cases)")

    def group(self, index, model, strategy):
        topology = topology_label(strategy)
        name = f"{index:03d}-{slug(model['name'])}__{slug(strategy['name'])}-{topology}"
        folder = self.run_dir / name
        folder.mkdir()
        result = dict(group=name, model=model["name"], strategy=strategy["name"],
                      started_at=timestamp(), state="failed", directory=str(folder))
        self.save_state(current_group=name, phase="starting")
        env = os.environ | self.c["env"] | model["env"] | strategy["env"]
        env.update(MODEL=model["path"], MODEL_NAME=model["name"],
                   TP=str(strategy["tp"]), PP=str(strategy["pp"]), DP=str(strategy["dp"]),
                   EP=str(strategy["ep"]).lower(), HOST=self.c["server"]["host"],
                   PORT=str(self.c["server"]["port"]), START_USE_DEFAULT_ARGS="0",
                   PYTHONUNBUFFERED="1")
        fatal = None
        with (folder / "group.log").open("a", buffering=1) as group_log, \
                (folder / "server.log").open("a", buffering=1) as server_log:
            self.group_log = group_log
            try:
                self.check_stop()
                self.assert_port_free()
                self.log(f"Starting group {name}")
                command = ["bash", str(HERE / "start.sh"), "--foreground"]
                command += self.c["serve_args"] + model["serve_args"] + strategy["serve_args"]
                server = self.launch(command, env, "server", server_log)
                self.save_state(server_pid=server.pid)
                self.wait_ready(server, model["name"])
                self.save_state(phase="benchmarking")
                b = self.c["benchmark"]
                env.update(MODEL=model["name"], TOKENIZER=model["path"],
                           STRATEGY_NAME=strategy["name"], BASE_URL=self.base_url,
                           RESULT_DIR=str(folder / "results"), LOG_DIR=str(folder / "cases"),
                           GROUP_LOG="", WORKLOADS_JSON=json.dumps(self.c["workloads"]),
                           CONCURRENCIES_JSON=json.dumps(self.c["concurrencies"]),
                           REQUEST_RATE=str(b["request_rate"]),
                           PROMPTS_MULTIPLIER=str(b["prompts_multiplier"]),
                           CASE_TIMEOUT=str(b["case_timeout"]),
                           CONTINUE_ON_ERROR=str(int(b["continue_on_error"])))
                bench = self.launch(["bash", str(HERE / "batch.sh")], env, "bench")
                self.save_state(benchmark_pid=bench.pid)
                failures = 0
                next_health = time.monotonic()
                while bench.poll() is None:
                    self.check_stop()
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited during benchmark: exit={server.returncode}")
                    if time.monotonic() >= next_health:
                        failures = 0 if self.healthy(model["name"]) else failures + 1
                        if failures >= self.c["server"]["health_failure_threshold"]:
                            raise RuntimeError("server health checks failed during benchmark")
                        next_health = time.monotonic() + self.c["server"]["health_interval"]
                    self.pause(0.2)
                self.check_stop()
                result["benchmark_exit_code"] = bench.returncode
                if bench.returncode != 0:
                    raise RuntimeError(f"batch.sh failed: exit={bench.returncode}; see group.log")
                if server.poll() is not None:
                    raise RuntimeError(f"server exited at benchmark completion: exit={server.returncode}")
                result["state"] = "completed"
                self.log("All cases completed; shutting down this group's server")
            except Stopped as exc:
                result.update(state="stopped", error=str(exc))
                fatal = exc
            except Exception as exc:
                result.update(state="failed", error=str(exc))
                self.log(f"Group failed: {exc}")
                if isinstance(exc, CleanupError):
                    fatal = exc
            finally:
                # Status/log errors must not prevent owned-process termination.
                try:
                    self.save_state(phase="stopping")
                except OSError as exc:
                    self.log_errors.append(str(exc))
                logs_drained = False
                try:
                    self.cleanup()
                    logs_drained = True
                    self.assert_port_free()
                    self.state.update(server_pid=None, benchmark_pid=None)
                except Exception as exc:
                    result.update(state="failed", error=f"cleanup failed: {exc}")
                    fatal = CleanupError(str(exc))
                # Export only after the producers and their log pumps have
                # finished, including failed/stopped groups with partial data.
                if logs_drained:
                    try:
                        self.export_group(folder, result, model, strategy)
                    except Exception as exc:
                        result.update(export_state="failed", export_error=str(exc))
                        if result["state"] == "completed":
                            result.update(state="failed", error=f"CSV export failed: {exc}")
                        if isinstance(exc, CleanupError):
                            fatal = exc
                        self.log(f"CSV export failed: {exc}")
                else:
                    result.update(export_state="skipped", export_error="log producers did not finish cleanup")
                if self.log_errors:
                    result.update(state="failed", error=f"output logging failed: {self.log_errors[0]}")
                    fatal = CleanupError(result["error"])
                self.group_log = None
                result["finished_at"] = timestamp()
                self.summary.append(result)
                write_json(folder / "status.json", result)
                write_json(self.run_dir / "summary.json", self.summary)
                self.save_state()
        self.log(f"Group {name}: {result['state']}")
        if fatal:
            raise fatal
        self.pause(self.c["server"]["settle_seconds"])
        return result["state"] == "completed"

    def run(self):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop_event.set())
        self.save_state()
        failed = False
        try:
            with open(self.c["lock_file"], "a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f"another matrix is running (lock: {self.c['lock_file']})") from exc
                self.assert_port_free()
                index = 0
                for model in self.c["models"]:
                    for strategy in self.c["strategies"]:
                        self.check_stop()
                        index += 1
                        success = self.group(index, model, strategy)
                        failed |= not success
                        if not success and not self.c["continue_on_error"]:
                            raise RuntimeError("group failed; continue_on_error=false")
        except Stopped as exc:
            self.save_state(state="stopped", error=str(exc), finished_at=timestamp())
            self.log("Matrix stopped; owned services and benchmarks cleaned up")
            return 130
        except Exception as exc:
            self.save_state(state="failed", error=str(exc), finished_at=timestamp())
            self.log(f"Matrix failed: {exc}")
            return 1
        self.save_state(state="failed" if failed else "completed", phase="finished",
                        finished_at=timestamp(), groups_attempted=len(self.summary),
                        groups_completed=sum(g["state"] == "completed" for g in self.summary),
                        groups_failed=sum(g["state"] == "failed" for g in self.summary))
        self.log(f"Matrix {self.state['state']}: {len(self.summary)} groups; {self.run_dir}")
        return int(failed)


def read_status(run_dir):
    state = json.loads((run_dir / "status.json").read_text())
    info = process_info(state.get("pid", 0))
    state["runner_alive"] = bool(info and info["state"] not in ("Z", "X")
                                 and info["ticks"] == state.get("proc_start_ticks"))
    if state["state"] not in TERMINAL and not state["runner_alive"]:
        state["state"] = "orphaned"
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("start", "run", "dry-run", "status", "stop"),
                        default="start", help="default: start using nohup in a new session")
    parser.add_argument("--config", type=Path, default=HERE / "matrix_config.json")
    parser.add_argument("--run-dir", type=Path, help="existing run for status/stop; new empty directory for run/start")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Linux is required for owned-session cleanup via /proc")
    try:
        if args.command in ("status", "stop"):
            if not args.run_dir:
                parser.error("--run-dir is required for status/stop")
            run_dir = args.run_dir.expanduser().resolve()
            state = read_status(run_dir)
            if args.command == "status":
                print(json.dumps(state, ensure_ascii=False, indent=2))
                return int(state["state"] in {"failed", "orphaned"})
            if state["state"] in TERMINAL:
                print(f"Already {state['state']}: {run_dir}")
                return 0
            if not state["runner_alive"]:
                raise RuntimeError("runner is gone; refusing to signal an unrelated/reused PID")
            os.kill(state["pid"], signal.SIGTERM)
            print(f"Stop requested for PID {state['pid']}; cleanup is asynchronous. "
                  f"Check status --run-dir {shlex.quote(str(run_dir))}")
            return 0
        c = load_config(args.config.expanduser().resolve())
        if args.command == "dry-run":
            total = len(c["models"]) * len(c["strategies"])
            cases = len(c["workloads"]) * len(c["concurrencies"])
            print(f"{total} groups, {cases} cases/group, {total * cases} cases total")
            print(json.dumps(c, ensure_ascii=False, indent=2))
            return 0
        if args.run_dir:
            run_dir = args.run_dir.expanduser().resolve()
            # Detached child consumes its parent's frozen config and reserved run directory.
            internal = args.command == "run" and os.environ.pop("OFFHAND_DETACHED_CHILD", "") == "1"
            if not internal:
                run_dir.mkdir(parents=True, exist_ok=True)
                if any(run_dir.iterdir()):
                    raise ValueError(f"run directory must be empty: {run_dir}")
        else:
            run_dir = Path(c["output_dir"]) / (dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                                               + f"-{os.getpid()}")
            run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", c)
        if args.command == "run":
            return Supervisor(c, run_dir).run()
        env = os.environ | {"OFFHAND_DETACHED_CHILD": "1", "PYTHONUNBUFFERED": "1"}
        command = ["nohup", sys.executable, str(Path(__file__).resolve()), "run", "--config",
                   str(run_dir / "config.json"), "--run-dir", str(run_dir)]
        with (run_dir / "runner.log").open("a") as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print(f"Run directory: {run_dir}\nRunner PID: {proc.pid}\n"
              f"Log: {run_dir / 'runner.log'}", flush=True)
        # Report immediate initialization failures without tying the job to this terminal.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (run_dir / "status.json").exists():
                state = read_status(run_dir)
                if state["state"] == "failed":
                    raise RuntimeError(f"startup failed: {state.get('error')}; see runner.log")
                return 0
            if proc.poll() is not None:
                raise RuntimeError(f"runner exited with {proc.returncode}; see runner.log")
            time.sleep(0.05)
        raise RuntimeError("runner did not acknowledge startup within 5s; inspect runner.log/status.json")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
