"""Read offhand matrix CSVs and legacy ``model_tpNppN.csv`` benchmark exports.

This module deliberately does not filter statuses, discard zero measurements,
aggregate repeated runs, or infer modern identities from archive filenames.
"""

from __future__ import annotations

import csv
import io
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Strategy:
    name: str
    tp: int
    pp: int
    dp: int
    ep_enabled: bool
    ep_size: int

    @property
    def label(self) -> str:
        topology = (
            f"TP{self.tp}/PP{self.pp}/DP{self.dp}/"
            f"EP{self.ep_size if self.ep_enabled else 'off'}"
        )
        return f"{self.name} ({topology})" if self.name else topology


@dataclass(frozen=True)
class Sample:
    model: str
    strategy: Strategy
    input_len: int
    output_len: int
    concurrency: int
    status: str
    metrics: dict[str, float | None]
    source: str
    row_number: int

    @property
    def workload(self) -> tuple[int, int]:
        return self.input_len, self.output_len


@dataclass(frozen=True)
class Metric:
    key: str
    column: str
    label: str
    unit: str
    lower_is_better: bool


METRICS = {
    "tpot": Metric("tpot", "非首token时延(ms)", "Mean TPOT", "ms/token", True),
    "ttft": Metric("ttft", "首token时延(ms)", "Mean TTFT", "ms", True),
    "throughput": Metric("throughput", "输出吞吐", "Output throughput", "tok/s", False),
    "per-concurrency": Metric(
        "per-concurrency", "单并发输出", "Output throughput / concurrency", "tok/s", False
    ),
    "duration": Metric("duration", "测试时间", "Benchmark duration", "s", True),
}

_WORKLOAD_COLUMNS = {"输入", "输出", "并发数"}
_REQUIRED_IDENTITY = {"模型", "并行策略", "TP", "PP", "DP", "EP_ENABLED"}
_IDENTITY_COLUMNS = _REQUIRED_IDENTITY | {"EP_SIZE"}
_STATUSES = {"completed", "failed", "incomplete"}
_LEGACY_FILENAME = re.compile(
    r"^(?P<model>.+?)_tp(?P<tp>\d+)pp(?P<pp>\d+)(?:_.*)?\.csv$",
    re.IGNORECASE,
)
# Legacy benchmark filenames use k = 1024 tokens, e.g. flash_tp1pp8_4k1k.csv.
_LEGACY_WORKLOAD = re.compile(r"_(\d+)k(\d+)k\.csv$", re.IGNORECASE)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _header(value: str) -> str:
    """Accept spreadsheet spacing such as ``首 token 时延 (ms)``."""
    return "".join(value.split())


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw.decode("gb18030")
        except UnicodeDecodeError as error:
            raise ValueError(f"{path}: expected UTF-8 or GB18030 CSV encoding") from error


def _positive_integer(value: str | None, column: str, location: str) -> int:
    try:
        result = int(value) if value is not None else 0
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{location}: {column} must be a positive integer (got {value!r})"
        ) from error
    if result <= 0:
        raise ValueError(
            f"{location}: {column} must be a positive integer (got {value!r})"
        )
    return result


def _required_text(row: dict, column: str, location: str) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(f"{location}: {column} must not be blank")
    return value


def _modern_identity(row: dict, location: str) -> tuple[str, Strategy]:
    model = _required_text(row, "模型", location)
    name = _required_text(row, "并行策略", location)
    tp, pp, dp = (
        _positive_integer(row.get(column), column, location)
        for column in ("TP", "PP", "DP")
    )
    ep_value = _required_text(row, "EP_ENABLED", location).lower()
    if ep_value not in {"0", "1", "true", "false"}:
        raise ValueError(
            f"{location}: EP_ENABLED must be 0, 1, true or false (got {ep_value!r})"
        )
    ep_enabled = ep_value in {"1", "true"}
    expected_size = tp * dp if ep_enabled else 1
    ep_size_value = (row.get("EP_SIZE") or "").strip()
    ep_size = (
        _positive_integer(ep_size_value, "EP_SIZE", location)
        if ep_size_value else expected_size
    )
    if ep_size != expected_size:
        raise ValueError(
            f"{location}: EP_SIZE={ep_size} disagrees with EP_ENABLED={ep_value}, "
            f"TP={tp}, DP={dp}; expected {expected_size} (TP * DP when enabled, else 1)"
        )
    return model, Strategy(name, tp, pp, dp, ep_enabled, ep_size)


def _legacy_identity(path: Path, strip_prefix: str) -> tuple[str, Strategy]:
    match = _LEGACY_FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"{path}: CSV without identity columns needs a legacy filename "
            "model_tpNppN[_suffix].csv; modern CSVs need 模型/并行策略/TP/PP/DP/EP_ENABLED"
        )
    model = match.group("model")
    if strip_prefix and model.startswith(strip_prefix):
        model = model[len(strip_prefix):]
    if not model.strip():
        raise ValueError(f"{path}: legacy model name is empty after prefix stripping")
    tp = _positive_integer(match.group("tp"), "TP in filename", str(path))
    pp = _positive_integer(match.group("pp"), "PP in filename", str(path))
    return model, Strategy("", tp, pp, 1, False, 1)


def _measurement(value: str | None, metric: Metric, location: str) -> float | None:
    try:
        result = float(value) if value is not None else math.nan
    except (TypeError, ValueError, OverflowError):
        result = math.nan
    if not math.isfinite(result) or result < 0:
        _warn(f"{location}: missing/invalid {metric.column} {value!r}; retained as empty")
        return None
    return result


def _load_file(
    path: Path, metrics: list[Metric], strip_prefix: str, explicit: bool
) -> list[Sample]:
    samples: list[Sample] = []
    try:
        with io.StringIO(_read_text(path), newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            original_headers = reader.fieldnames or []
            headers = [_header(value) for value in original_headers]
            header_set = set(headers)
            modern = bool(header_set & _IDENTITY_COLUMNS)
            recognized = modern or bool(_LEGACY_FILENAME.fullmatch(path.name)) or (
                _WORKLOAD_COLUMNS <= header_set
            )
            if not recognized:
                message = f"{path}: unsupported CSV (no offhand identity or legacy benchmark schema)"
                if explicit:
                    raise ValueError(message)
                _warn(f"skip {message}")
                return samples
            if len(header_set) != len(headers):
                raise ValueError(f"{path}: duplicate CSV columns after whitespace normalization")
            required = _WORKLOAD_COLUMNS | {_header(metric.column) for metric in metrics}
            if modern:
                required |= _REQUIRED_IDENTITY
            missing = required - header_set
            if missing:
                raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")
            reader.fieldnames = headers
            identity = None if modern else _legacy_identity(path, strip_prefix)
            workload_claim = None if modern else _LEGACY_WORKLOAD.search(path.name)
            expected_workload = (
                tuple(int(value) * 1024 for value in workload_claim.groups())
                if workload_claim else None
            )
            while True:
                # csv.line_num also handles quoted multiline fields correctly.
                row_number = reader.line_num + 1
                try:
                    row = next(reader)
                except StopIteration:
                    break
                location = f"{path}:{row_number}"
                if None in row:
                    raise ValueError(f"{location}: more CSV cells than header columns")
                model, strategy = _modern_identity(row, location) if modern else identity
                input_len = _positive_integer(row.get("输入"), "输入", location)
                output_len = _positive_integer(row.get("输出"), "输出", location)
                concurrency = _positive_integer(row.get("并发数"), "并发数", location)
                if expected_workload and (input_len, output_len) != expected_workload:
                    raise ValueError(
                        f"{location}: workload mismatch: filename declares "
                        f"{expected_workload[0]}/{expected_workload[1]} tokens, row contains "
                        f"{input_len}/{output_len}; correct the data or filename"
                    )
                status = "legacy"
                if "状态" in header_set:
                    status = (row.get("状态") or "").strip()
                    if status not in _STATUSES:
                        raise ValueError(
                            f"{location}: 状态 must be completed, failed or incomplete "
                            f"(got {status!r}); blank/unknown status cannot imply success"
                        )
                samples.append(Sample(
                    model=model,
                    strategy=strategy,
                    input_len=input_len,
                    output_len=output_len,
                    concurrency=concurrency,
                    status=status,
                    metrics={
                        metric.key: _measurement(row.get(_header(metric.column)), metric, location)
                        for metric in metrics
                    },
                    source=str(path.resolve()),
                    row_number=row_number,
                ))
    except csv.Error as error:
        raise ValueError(f"{path}: invalid CSV: {error}") from error
    return samples


def load_samples(
    path: Path,
    metrics: list[Metric],
    file_glob: str = "*.csv",
    legacy_strip_prefix: str = "",
) -> list[Sample]:
    """Load individual rows from one CSV or recursively from a result directory.

    Any identity column switches on strict modern identity validation. Only CSVs
    with no identity columns may use a legacy filename. Status ``legacy`` means
    that the source has no status column and its success has not been verified.
    Unknown CSV schemas in directories are skipped with a warning; recognized
    benchmark files with missing or malformed required fields always fail.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"{path}: expected a .csv file or a directory")
        files = [path]
        explicit = True
    elif path.is_dir():
        try:
            files = sorted(candidate for candidate in path.rglob(file_glob) if candidate.is_file())
        except (OSError, ValueError, NotImplementedError) as error:
            raise ValueError(f"{path}: cannot enumerate glob {file_glob!r}: {error}") from error
        explicit = False
    else:
        raise ValueError(f"input CSV or directory does not exist: {path}")
    samples = []
    for csv_path in files:
        samples.extend(_load_file(csv_path, metrics, legacy_strip_prefix, explicit))
    if not samples:
        raise ValueError(f"{path}: no benchmark rows found (glob {file_glob!r})")
    return samples
