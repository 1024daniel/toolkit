#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plotter="$script_dir/vllm/benchmark_plots/plot_benchmarks.py"

nvlink_run="$script_dir/offhand_batch_test/offhand_runs/20260916-055558-150568-489629"
pcie_run="$script_dir/offhand_batch_test/offhand_runs/pcie-20260916-114055-008603-790373"

output_dir="$script_dir/dsv4pro_benchmark_compare"
metrics=(--metric tpot --metric throughput)

# Compare nvlink and pcie for each parallel strategy.
python3 "$plotter" \
  --dataset "nvlink=$nvlink_run" \
  --dataset "pcie=$pcie_run" \
  "${metrics[@]}" \
  --output-dir "$output_dir"

# Compare parallel strategies within each environment.
python3 "$plotter" \
  --dataset "nvlink=$nvlink_run" \
  "${metrics[@]}" \
  --output-dir "$output_dir/nvlink"

python3 "$plotter" \
  --dataset "pcie=$pcie_run" \
  "${metrics[@]}" \
  --output-dir "$output_dir/pcie"
