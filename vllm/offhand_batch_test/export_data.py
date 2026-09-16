#!/usr/bin/env python3
"""Export legacy benchmark logs or offhand group logs to an Excel-friendly CSV."""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


HEADERS = [
    "输入", "输出", "并发数", "单并发输出", "输出吞吐", "首token时延(ms)",
    "非首token时延(ms)", "测试时间", "测试时间段",
]
MANAGED_HEADERS = ["状态", "用例", "模型", "并行策略", "TP", "PP", "DP", "EP_ENABLED", "EP_SIZE"]
IDENTITY_FIELDS = ("model", "strategy", "tp", "pp", "dp", "ep")
TOPOLOGY = re.compile(r"^TP=(\d+)\s+PP=(\d+)\s+DP=(\d+)\s+EP=(true|false|1|0)$", re.IGNORECASE)
STRATEGY = re.compile(
    r"^Strategy:\s*(.*?)\s+\((TP=\d+\s+PP=\d+\s+DP=\d+\s+EP=(?:true|false|1|0))\)$",
    re.IGNORECASE,
)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CASE_START = re.compile(r"^\[\d+/\d+\]\s+Start benchmark:\s*(.+)$")
METADATA = re.compile(
    r"^Input:\s*(\d+)\s+Output:\s*(\d+)\s+Concurrency:\s*(\d+)\s+Prompts:\s*(\d+)"
)
PARAMETERS = re.compile(
    r"\b(random_input_len|random_output_len|max_concurrency|num_prompts)"
    r"\s*=\s*(\d+|None)(?=\s*[,\)])"
)
TIMESTAMP = re.compile(r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
TIME_FORMAT = "%m-%d %H:%M:%S"
METRICS = {
    "Maximum request concurrency": "max_concurrency",
    "Successful requests": "successful_requests",
    "Benchmark duration (s)": "benchmark_duration",
    "Total input tokens": "total_input_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Request throughput (req/s)": "request_throughput",
    "Output token throughput (tok/s)": "output_token_throughput",
    "Total Token throughput (tok/s)": "total_token_throughput",
    "Mean TTFT (ms)": "mean_TTFT",
    "Median TTFT (ms)": "median_TTFT",
    "P99 TTFT (ms)": "p99_TTFT",
    "Mean TPOT (ms)": "mean_TPOT",
    "Median TPOT (ms)": "median_TPOT",
    "P99 TPOT (ms)": "p99_TPOT",
    "Mean ITL (ms)": "mean_ITL",
    "Median ITL (ms)": "median_ITL",
    "P99 ITL (ms)": "p99_ITL",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()[:10]


def filename_label(value, limit):
    """Keep safe labels readable, and distinguish sanitized/truncated names."""
    value = str(value)
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "unnamed"
    if label != value or len(label) > limit:
        label = label[:limit - 11].rstrip("._-") + "-" + digest(value)
    return label


def ep_size(topology):
    """EP spans TP × DP ranks; disabled EP has a singleton group."""
    if topology.get("ep") is None:
        return None
    if not topology["ep"]:
        return 1
    tp, dp = topology.get("tp"), topology.get("dp")
    return int(tp) * int(dp) if tp is not None and dp is not None else None


def topology_label(topology):
    base = "-".join(f"{key}{int(topology[key])}" for key in ("tp", "pp", "dp"))
    return f"{base}-ep{ep_size(topology) if topology['ep'] else 'off'}"


def archive_filename(model, strategy, topology, workloads, concurrencies, run_tag):
    """Describe a matrix in a portable filename, bounded for atomic CSV writes.

    Input/output pairs stay paired. Long matrices retain a readable summary and
    a digest of the complete ordered list; raw identities also live in CSV rows.
    """
    pairs = list(dict.fromkeys(tuple(pair) for pair in workloads))
    levels = list(dict.fromkeys(concurrencies))
    parallel_label = topology_label(topology)
    io_label = "+".join(f"in{input_len}-out{output_len}" for input_len, output_len in pairs) or "io-none"
    concurrency_label = "c" + "-".join(map(str, levels)) if levels else "c-none"

    def assemble(limits=(64, 48, 48)):
        labels = [filename_label(value, limit)
                  for value, limit in zip((model, strategy, run_tag), limits)]
        return (f"{labels[0]}__{labels[1]}-{parallel_label}__{io_label}"
                f"__{concurrency_label}__{labels[2]}.csv")

    if len(assemble()) > 240 and pairs:
        first = f"in{pairs[0][0]}-out{pairs[0][1]}"
        compact = f"{first}+more{len(pairs) - 1}io-{digest(pairs)}"
        if len(compact) < len(io_label):
            io_label = compact
    if len(assemble()) > 240 and levels:
        compact = f"c{min(levels)}-{max(levels)}-n{len(levels)}-{digest(levels)}"
        if len(compact) < len(concurrency_label):
            concurrency_label = compact
    filename = assemble()
    if len(filename) > 240:
        filename = assemble((32, 24, 32))
    if len(filename) > 240:
        # Also bound unusually large numbers from manually authored logs.
        io_label = filename_label(io_label, 48)
        concurrency_label = filename_label(concurrency_label, 40)
        filename = assemble((24, 20, 24))
    if len(filename) > 240:
        raise ValueError("parallel topology is too long for an archive filename")
    return filename


def identity_for(line):
    """Read only explicit batch metadata, after filtering log-source prefixes."""
    if line.startswith("Model:"):
        return {"model": line.partition(":")[2].strip()}
    strategy = STRATEGY.match(line)
    metadata = {}
    if strategy:
        metadata["strategy"] = strategy.group(1)
        line = strategy.group(2)
    elif line.startswith("Strategy:"):
        return {"strategy": line.partition(":")[2].strip()}
    match = TOPOLOGY.match(line)
    if match:
        tp, pp, dp, ep = match.groups()
        metadata.update(tp=int(tp), pp=int(pp), dp=int(dp), ep=int(ep.lower() in ("true", "1")))
    return metadata


def number(value):
    """Missing, invalid and non-finite measurements become empty CSV cells."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def integer(value):
    result = number(value)
    return int(result) if result is not None and result.is_integer() else None


def row_for(data):
    concurrency = integer(data.get("max_concurrency"))
    throughput = number(data.get("output_token_throughput"))
    per_concurrency = throughput / concurrency if throughput is not None and concurrency else None
    time_range = ""
    try:
        # The logs omit a year. A fixed leap year also supports February 29 and
        # avoids Python's deprecated implicit year when parsing month/day dates.
        start = datetime.strptime("2000-" + data["start_time"], "%Y-" + TIME_FORMAT)
        end = datetime.strptime("2000-" + data["last_log_time"], "%Y-" + TIME_FORMAT)
        end += timedelta(seconds=float(data["benchmark_duration"]))
        time_range = f"{start.strftime(TIME_FORMAT)}-{end.strftime(TIME_FORMAT)}"
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return [
        integer(data.get("random_input_len")), integer(data.get("random_output_len")),
        concurrency, per_concurrency, throughput,
        *[data.get(key) if number(data.get(key)) is not None else None
          for key in ("mean_TTFT", "mean_TPOT", "benchmark_duration")],
        time_range,
    ]


def parse_log(stream):
    """Keep each managed case until its next start marker or EOF, even if failed.

    vLLM percentile ordering is not a case boundary. Only legacy logs retain the
    historical P99 ITL boundary, since those logs have no batch case markers.
    """
    records = []
    data = {}
    group_identity = {}
    managed = False
    has_metrics = False

    def flush():
        if data.get("case") is not None or (not managed and has_metrics):
            records.append(dict(data))

    for raw in stream:
        line = ANSI.sub("", raw).strip()
        if line.startswith("[bench]"):
            line = line[len("[bench]"):].strip()
        elif line.startswith("[") and not CASE_START.match(line):
            # Server, scheduler and exporter output share group.log. None of
            # them may contribute benchmark metadata, metrics or timestamps.
            continue

        match = CASE_START.match(line)
        if match:
            if not managed:
                records.clear()
                managed = True
            else:
                flush()
            data = {**group_identity, "case": match.group(1), "status": "incomplete"}
            has_metrics = False
            continue
        metadata = identity_for(line)
        if metadata:
            if not managed:
                group_identity.update(metadata)
            elif data.get("case"):
                data.update(metadata)
            continue
        if managed and not data.get("case"):
            continue

        match = METADATA.match(line)
        if match:
            data.update(zip(
                ("random_input_len", "random_output_len", "max_concurrency", "num_prompts"),
                map(int, match.groups()),
            ))
        elif line.startswith("Namespace("):
            if not managed and has_metrics:
                flush()
                data = {}
                has_metrics = False
            for key, value in PARAMETERS.findall(line):
                if value != "None" and (not managed or data.get(key) is None):
                    data[key] = int(value)
        elif line.startswith("INFO"):
            match = TIMESTAMP.search(line)
            if match:
                data.setdefault("start_time", match.group())
                data["last_log_time"] = match.group()
        elif managed and line.startswith("SUCCESS:"):
            data["status"] = "completed"
        elif managed and line.startswith("FAILURE:"):
            data["status"] = "failed"
        else:
            label, separator, value = line.partition(":")
            key = METRICS.get(label)
            if separator and key:
                data[key] = value.strip() or None
                has_metrics = True
                if not managed and key == "p99_ITL":
                    flush()
                    data = {}
                    has_metrics = False
    flush()
    rows = [row_for(record) + ([record["status"], record["case"],
                               *[record.get(key) for key in IDENTITY_FIELDS], ep_size(record)] if managed else [])
            for record in records]
    rows.sort(key=lambda row: tuple((value is None, value or 0) for value in row[:3]))
    return HEADERS + (MANAGED_HEADERS if managed else []), rows


def default_output(log_file, headers, rows):
    """Use row metadata for managed logs; retain legacy log.csv behavior."""
    if headers != HEADERS + MANAGED_HEADERS or not rows:
        return log_file.with_suffix(".csv")
    identities = {tuple(row[11:17]) for row in rows}
    if len(identities) != 1:
        return log_file.with_suffix(".csv")
    model, strategy, tp, pp, dp, ep = identities.pop()
    if not model or not strategy or None in (tp, pp, dp, ep):
        return log_file.with_suffix(".csv")
    if any(None in row[:3] for row in rows):
        return log_file.with_suffix(".csv")
    timestamp = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y%m%d-%H%M%S-%f")
    return log_file.parent / archive_filename(
        model, strategy, dict(zip(("tp", "pp", "dp", "ep"), (tp, pp, dp, ep))),
        [row[:2] for row in rows], [row[2] for row in rows], timestamp,
    )


def write_csv(path, headers, rows):
    """Publish only a complete CSV; preserve any previous archive on failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="benchmark log or offhand group.log")
    parser.add_argument("--output", type=Path,
                        help="CSV destination (default: managed matrix filename, or legacy log.csv)")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.resolve() == args.log_file.resolve():
            raise ValueError("CSV destination must differ from the input log")
        with args.log_file.open(encoding="utf-8", errors="replace") as stream:
            headers, rows = parse_log(stream)
        if not rows:
            raise ValueError("no benchmark cases found in the log")
        output = args.output if args.output is not None else default_output(args.log_file, headers, rows)
        if output.resolve() == args.log_file.resolve():
            raise ValueError("CSV destination must differ from the input log")
        write_csv(output, headers, rows)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: CSV export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {len(rows)} cases to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
