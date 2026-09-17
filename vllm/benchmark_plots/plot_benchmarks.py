#!/usr/bin/env python3
"""Plot and compare offhand benchmark CSVs using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from benchmark_data import METRICS, Metric, Sample, Strategy, load_samples
from svg_charts import render_heatmap, render_lines


@dataclass(frozen=True, order=True)
class Key:
    model: str
    strategy: Strategy
    input_len: int
    output_len: int
    concurrency: int


@dataclass
class Point:
    value: float | None
    status: str
    samples: list[Sample]
    n_valid: int


DIMENSIONS = ["model", "strategy", "tp", "pp", "dp", "ep_enabled", "ep_size",
              "input_len", "output_len", "concurrency"]


def dimensions(key: Key) -> dict:
    s = key.strategy
    return dict(zip(DIMENSIONS, [key.model, s.name, s.tp, s.pp, s.dp,
                                int(s.ep_enabled), s.ep_size, key.input_len,
                                key.output_len, key.concurrency]))


def key_for(sample: Sample, match_strategy: str) -> Key:
    strategy = sample.strategy
    if match_strategy == "topology":
        strategy = replace(strategy, name="")
    return Key(sample.model, strategy, sample.input_len, sample.output_len,
               sample.concurrency)


def measurement_status(sample: Sample, metric: Metric, include_zero: bool) -> str:
    if sample.status not in ("completed", "legacy"):
        return "failed"
    value = sample.metrics[metric.key]
    if value is None or (value == 0 and not include_zero):
        return "invalid"
    return "ok"


def aggregate(samples: list[Sample], metric: Metric, args: argparse.Namespace) -> dict[Key, Point]:
    groups: dict[Key, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[key_for(sample, args.match_strategy)].append(sample)
    points = {}
    for key, rows in sorted(groups.items()):
        if len(rows) > 1 and args.aggregate == "error":
            sources = ", ".join(f"{s.source}:{s.row_number}" for s in rows)
            raise ValueError(
                f"duplicate sample: {key.model} {key.strategy.label}, "
                f"input/output={key.input_len}/{key.output_len}, C={key.concurrency}: "
                f"{sources}; narrow --glob/--strategy or explicitly set --aggregate"
            )
        statuses = [measurement_status(s, metric, args.include_zero) for s in rows]
        values = [s.metrics[metric.key] for s, status in zip(rows, statuses) if status == "ok"]
        if values:
            reducers = {"error": statistics.fmean, "mean": statistics.fmean,
                        "median": statistics.median, "min": min, "max": max}
            value = reducers[args.aggregate](values)
            if not math.isfinite(value):
                raise ValueError(f"non-finite aggregate for {key}")
            status = "ok" if len(values) == len(rows) else "partial"
        else:
            value = None
            status = "failed" if "failed" in statuses else "invalid"
        points[key] = Point(value, status, rows, len(values))
    return points


def comparison(baseline: Point | None, sample: Point | None) -> dict:
    result = dict(baseline_value=baseline.value if baseline else None,
                  value=sample.value if sample else None,
                  baseline_status=baseline.status if baseline else "missing",
                  sample_status=sample.status if sample else "missing",
                  baseline_n_valid=baseline.n_valid if baseline else 0,
                  n_valid=sample.n_valid if sample else 0,
                  ratio=None, delta=None, change_pct=None)
    if baseline is None or sample is None:
        result["status"] = "missing"
    elif baseline.value is None or sample.value is None:
        result["status"] = "failed" if "failed" in (baseline.status, sample.status) else "invalid"
    else:
        delta = sample.value - baseline.value
        result["delta"] = delta
        if baseline.value == 0:
            result["status"] = "zero_baseline"
        else:
            ratio = sample.value / baseline.value
            change = (ratio - 1) * 100
            if not all(math.isfinite(v) for v in (delta, ratio, change)):
                result.update(status="invalid", delta=None)
            else:
                result.update(status="ok", ratio=ratio, change_pct=change)
    return result


def assignment(value: str, option: str) -> tuple[str, str]:
    label, separator, content = value.partition("=")
    if not separator or not label.strip() or not content:
        raise ValueError(f"{option} expects LABEL=VALUE, got {value!r}")
    return label.strip(), content


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot offhand CSVs: one run with strategy curves, or multiple named runs "
                    "with baseline ratio heatmaps. Each workload gets a separate figure.")
    parser.add_argument("path", nargs="?", type=Path, help="one CSV, group directory or run directory")
    parser.add_argument("--dataset", action="append", default=[], metavar="LABEL=PATH",
                        help="named CSV/directory; repeat for comparisons (exclusive with positional path)")
    parser.add_argument("--baseline", help="baseline dataset label (default: first --dataset)")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_plots_output"))
    parser.add_argument("--metric", action="append", choices=[*METRICS, "all"],
                        help="repeat for multiple metrics; default: tpot")
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--model", action="append", help="original or matched model name; repeat to select more")
    parser.add_argument("--match-model", choices=("exact", "prefix"), default="prefix",
                        help="match optional environment prefixes against baseline models (default: prefix); "
                             "use exact to require identical names")
    parser.add_argument("--strategy", action="append", help="exact strategy name; repeat to select more")
    parser.add_argument("--match-strategy", choices=("name", "topology"), default="name",
                        help="match name+TP/PP/DP/EP (default), or topology only")
    parser.add_argument("--aggregate", choices=("error", "mean", "median", "min", "max"),
                        default="error", help="duplicate handling; explicit aggregation uses valid samples only")
    parser.add_argument("--glob", default="*.csv", help="recursive directory CSV glob (default: *.csv)")
    parser.add_argument("--legacy-strip-prefix", action="append", default=[], metavar="LABEL=PREFIX",
                        help="strip this prefix from legacy filename-derived model names for the dataset")
    parser.add_argument("--include-zero", action="store_true", help="allow zero metrics in successful rows")
    parser.add_argument("--linear-x", action="store_true", help="linear concurrency axis (default: log2)")
    parser.add_argument("--y-scale", choices=("auto", "shared", "independent"), default="auto")
    parser.add_argument("--value-labels", choices=("all", "last", "none"), default="last")
    parser.add_argument("--columns", type=int, default=2, help="maximum line-chart panels per row")
    parser.add_argument("--title", help="figure title prefix")
    return parser.parse_args(argv)


def align_models(datasets: dict[str, list[Sample]], baseline: str) -> dict[str, dict[str, str]]:
    """Match complete baseline names, allowing separator-delimited prefixes only."""
    baseline_models = {s.model for s in datasets[baseline]}

    def prefixed(longer: str, shorter: str) -> bool:
        return any(longer.endswith(separator + shorter) for separator in ("-", "_", " "))

    mappings = {}
    for label, samples in datasets.items():
        mapping = {}
        for model in sorted({s.model for s in samples}):
            matches = ([model] if model in baseline_models else
                       sorted(name for name in baseline_models
                              if prefixed(model, name) or prefixed(name, model)))
            if len(matches) > 1:
                raise ValueError(f"dataset {label!r}: ambiguous model match for {model!r}: "
                                 f"{matches}; use --match-model exact")
            mapping[model] = matches[0] if matches else model
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"dataset {label!r}: model prefix matching would merge distinct "
                             "models in the same dataset; use --match-model exact")
        mappings[label] = mapping
    return mappings


def load_datasets(args: argparse.Namespace, metrics: list[Metric]) -> dict[str, list[Sample]]:
    if args.path is not None and args.dataset:
        raise ValueError("use a positional path or --dataset, not both")
    if args.path is None and not args.dataset:
        raise ValueError("provide a CSV/directory path or at least one --dataset LABEL=PATH")
    if (args.input_len is None) != (args.output_len is None):
        raise ValueError("--input-len and --output-len must be specified together")
    if args.input_len is not None and min(args.input_len, args.output_len) <= 0:
        raise ValueError("input/output lengths must be positive")
    if args.columns <= 0:
        raise ValueError("--columns must be positive")
    specs = ([((args.path.stem if args.path.is_file() else args.path.name) or "data", str(args.path))]
             if args.path is not None else [assignment(v, "--dataset") for v in args.dataset])
    if len({label for label, _ in specs}) != len(specs):
        raise ValueError("dataset labels must be unique")
    prefixes = {}
    for spec in args.legacy_strip_prefix:
        label, prefix = assignment(spec, "--legacy-strip-prefix")
        if label not in {item[0] for item in specs} or label in prefixes:
            raise ValueError(f"unknown or repeated dataset in --legacy-strip-prefix: {label}")
        prefixes[label] = prefix
    args.baseline = args.baseline or specs[0][0]
    if args.baseline not in {item[0] for item in specs}:
        raise ValueError(f"unknown baseline label: {args.baseline}")
    datasets = {}
    args.source_paths = set()
    for label, path in specs:
        samples = load_samples(Path(path), metrics, args.glob, prefixes.get(label, ""))
        args.source_paths.update(Path(s.source).resolve() for s in samples)
        datasets[label] = samples
    mappings = (align_models(datasets, args.baseline) if args.match_model == "prefix" else
                {label: {s.model: s.model for s in samples} for label, samples in datasets.items()})
    for label, samples in datasets.items():
        mapping = mappings[label]
        for original, matched in mapping.items():
            if original != matched:
                print(f"model match {label}: {original!r} -> {matched!r}")
        selected = [replace(s, model=mapping[s.model]) for s in samples
                    if (args.model is None or s.model in args.model or mapping[s.model] in args.model)
                    and (args.strategy is None or s.strategy.name in args.strategy)
                    and (args.input_len is None or s.workload == (args.input_len, args.output_len))]
        if not selected:
            raise ValueError(f"dataset {label!r}: no samples remain after filtering; "
                             f"available workloads: {sorted({s.workload for s in samples})}")
        datasets[label] = selected
        print(f"loaded {label}: {len(selected)} rows, "
              f"{len({s.source for s in selected})} CSVs; states={dict(Counter(s.status for s in selected))}")
    return datasets


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def panels_for(data: dict[str, dict[Key, Point]], workload: tuple[int, int]) -> list[dict]:
    keys = sorted({k for points in data.values() for k in points
                   if (k.input_len, k.output_len) == workload})
    concurrencies = sorted({k.concurrency for k in keys})
    panels = []
    if len(data) == 1:
        points = next(iter(data.values()))
        for model in sorted({k.model for k in keys}):
            series = []
            for strategy in sorted({k.strategy for k in keys if k.model == model}):
                values = {}
                for c in concurrencies:
                    point = points.get(Key(model, strategy, *workload, c))
                    values[c] = point.value if point else None
                series.append(dict(label=strategy.label, points=values,
                                   style_group=(strategy.tp, strategy.pp, strategy.dp),
                                   ep_enabled=strategy.ep_enabled,
                                   end_label=f"TP{strategy.tp}/PP{strategy.pp}/DP{strategy.dp} · "
                                             f"EP{strategy.ep_size if strategy.ep_enabled else 'off'}"))
            panels.append(dict(title=model, series=series))
    else:
        for model, strategy in sorted({(k.model, k.strategy) for k in keys}):
            series = []
            for label, points in data.items():
                values = {}
                for c in concurrencies:
                    point = points.get(Key(model, strategy, *workload, c))
                    values[c] = point.value if point else None
                series.append(dict(label=label, points=values))
            panels.append(dict(title=f"{model} · {strategy.label}", series=series))
    return panels


def run(args: argparse.Namespace) -> None:
    names = list(dict.fromkeys(args.metric or ["tpot"]))
    if "all" in names:
        names = list(METRICS)
    metrics = [METRICS[name] for name in names]
    datasets = load_datasets(args, metrics)
    workloads = sorted({s.workload for samples in datasets.values() for s in samples})
    all_data = {m.key: {label: aggregate(samples, m, args) for label, samples in datasets.items()}
                for m in metrics}
    sample_rows, summary_rows, comparison_rows = [], [], []
    charts = []
    for metric in metrics:
        data = all_data[metric.key]
        for label, samples in datasets.items():
            for sample in samples:
                sample_rows.append(dict(dataset=label, **dimensions(key_for(sample, "name")),
                                        metric=metric.key, value=sample.metrics[metric.key],
                                        status=sample.status,
                                        measurement_status=measurement_status(sample, metric, args.include_zero),
                                        source=sample.source, row_number=sample.row_number))
            for key, point in sorted(data[label].items()):
                summary_rows.append(dict(dataset=label, **dimensions(key), metric=metric.key,
                                         value=point.value, status=point.status,
                                         n_total=len(point.samples), n_valid=point.n_valid,
                                         sources=json.dumps([f"{s.source}:{s.row_number}" for s in point.samples],
                                                            ensure_ascii=False)))
            excluded = sum(measurement_status(s, metric, args.include_zero) != "ok" for s in samples)
            if excluded:
                print(f"warning: {label}/{metric.key}: {excluded}/{len(samples)} rows excluded "
                      "from numeric curves and aggregation; see samples.csv", file=sys.stderr)
        baseline = data[args.baseline]
        heatmaps: dict[tuple[int, int], dict[tuple[str, str, Strategy], dict]] = defaultdict(dict)
        for label, points in data.items():
            if label == args.baseline:
                continue
            matched = 0
            for key in sorted(baseline.keys() | points.keys()):
                result = comparison(baseline.get(key), points.get(key))
                matched += result["status"] == "ok"
                comparison_rows.append(dict(baseline=args.baseline, dataset=label, **dimensions(key),
                                            metric=metric.key, **result))
                workload = (key.input_len, key.output_len)
                row = heatmaps[workload].setdefault((label, key.model, key.strategy),
                    dict(label=f"{label} / {args.baseline} · {key.model} · {key.strategy.label}", cells={}))
                detail = (f"{label}={result['value']}, {args.baseline}={result['baseline_value']} "
                          f"{metric.unit}; status={result['status']}; "
                          f"sample={result['sample_status']}, baseline={result['baseline_status']}; "
                          f"valid n={result['n_valid']}/{result['baseline_n_valid']}")
                row["cells"][key.concurrency] = dict(ratio=result["ratio"], status=result["status"], detail=detail)
            if not matched:
                print(f"warning: {label}/{metric.key}: no valid matched pairs with baseline "
                      f"{args.baseline!r}; check model/strategy/workload identities", file=sys.stderr)
        for workload in workloads:
            stem = f"{metric.key}_in{workload[0]}_out{workload[1]}"
            title = f"{args.title + ' · ' if args.title else ''}{metric.label} vs. Concurrency"
            direction = "lower is better" if metric.lower_is_better else "higher is better"
            reduction_note = "Duplicates: reject" if args.aggregate == "error" else f"Aggregate: {args.aggregate}"
            match_note = "name + topology" if args.match_strategy == "name" else "topology"
            subtitle = (f"Input {workload[0]} / Output {workload[1]} tokens · {direction} · "
                        f"{reduction_note} · Strategy match: {match_note}")
            if len(datasets) == 1:
                title += f" · {next(iter(datasets))}"
            charts.append(("lines", args.output_dir / f"{stem}.svg",
                           dict(panels=panels_for(data, workload), title=title, subtitle=subtitle,
                                y_label=f"{metric.label} ({metric.unit})", linear_x=args.linear_x,
                                y_scale=args.y_scale, value_labels=args.value_labels, columns=args.columns)))
            if len(data) > 1:
                charts.append(("heatmap", args.output_dir / f"{stem}_ratio.svg",
                               dict(rows=list(heatmaps[workload].values()),
                                    title=f"{metric.label}: candidate / {args.baseline}",
                                    subtitle=subtitle, lower_is_better=metric.lower_is_better)))
    csv_names = ["samples.csv", "summary.csv"] + (["comparison.csv"] if len(datasets) > 1 else [])
    targets = [path for _, path, _ in charts] + [args.output_dir / name for name in csv_names]
    for path in targets:
        if path.resolve() in args.source_paths or (
            path.exists() and any(path.samefile(source) for source in args.source_paths)
        ):
            raise ValueError(f"output would overwrite input CSV: {path}; choose another --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind, path, options in charts:
        renderer = render_lines if kind == "lines" else render_heatmap
        renderer(output=path, **options)
        print(f"saved {path}")
    write_csv(args.output_dir / "samples.csv", ["dataset", *DIMENSIONS, "metric", "value", "status",
                                              "measurement_status", "source", "row_number"], sample_rows)
    write_csv(args.output_dir / "summary.csv", ["dataset", *DIMENSIONS, "metric", "value", "status",
                                              "n_total", "n_valid", "sources"], summary_rows)
    if len(datasets) > 1:
        write_csv(args.output_dir / "comparison.csv",
                  ["baseline", "dataset", *DIMENSIONS, "metric", "baseline_value", "value", "ratio",
                   "delta", "change_pct", "status", "baseline_status", "sample_status",
                   "baseline_n_valid", "n_valid"], comparison_rows)
    print(f"saved {', '.join(csv_names)} in {args.output_dir}; "
          f"{len(datasets)} datasets, {len(workloads)} workloads, {len(metrics)} metrics")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, csv.Error, OverflowError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
