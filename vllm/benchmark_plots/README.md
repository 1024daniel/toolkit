# offhand 通用性能对比绘图

将原 `plot_tpot.py` 的“单组数据中比较并行策略”和 `compare_tpot.py` 的“多组数据比较 + 比值热力图”合并为一个入口，适配本项目 [offhand_batch_test](../offhand_batch_test/README.md) 的归档 CSV。支持任意模型名、多组运行结果、TP/PP/DP/EP、多种指标和多个输入输出组合。

只依赖 **Python 3.9+ 标准库**，无需安装 matplotlib、pandas 或 vLLM。生成可直接在浏览器打开、放大或插入文档的 SVG，以及 UTF-8 BOM 编码的 CSV。整个目录可以单独复制使用，不依赖原 `prof_data` 目录。

压测时的环境变量打印仍由 [offhand_batch_test](../offhand_batch_test/README.md) 的调度与启动脚本记录在日志中，绘图工具读取归档 CSV；本次绘图优化不改变环境记录行为。

## 快速使用

以下命令均从本仓库根目录执行；也可以将脚本路径替换为绝对路径。

### 一次运行内比较并行策略

```bash
python3 vllm/benchmark_plots/plot_benchmarks.py \
  /path/to/offhand_runs/20260917-120000-000000-12345 \
  --output-dir /path/to/plots/single
```

递归读取 run 目录内每组 CSV，每个模型一个子图，每条线是一种策略。单数据集使用宽幅单列布局，同一 TP/PP/DP 的曲线保持相同颜色和点形，通过实线/虚线及实心/空心区分 EP 开关。也支持直接传入某个组目录、单个 CSV 或集中存放归档 CSV 的目录。默认绘制 TPOT，横轴为并发数的 log2 轴。

### 比较不同环境或不同版本

```bash
python3 vllm/benchmark_plots/plot_benchmarks.py \
  --dataset 'NVLink=/path/to/nvlink_run' \
  --dataset 'PCIe=/path/to/pcie_run' \
  --dataset 'New version=/path/to/new_version_run' \
  --baseline NVLink \
  --metric tpot --metric throughput \
  --output-dir /path/to/plots/compare
```

`--dataset` 可以重复任意次数，标签由用户定义，默认以第一个数据集为基线。单个位置参数与 `--dataset` 互斥。多数据集时，每个“模型 × 策略”一个子图，每条线是一个数据集；每个候选数据集与基线的比较合并在对应指标的热力图内。

CSV 必须来自可比较的压测设置。程序校验模型、并行策略和输入输出长度；CSV 未记录的 GPU、软件版本、请求速率、请求总数及启动参数，需要自行通过各 run 的 `config.json` 和日志确认。

### 指定输入输出、模型或策略

```bash
python3 vllm/benchmark_plots/plot_benchmarks.py \
  /path/to/run \
  --input-len 4096 --output-len 1024 \
  --model DeepSeek-V4-Flash \
  --strategy tp8 --strategy tp4_pp2 \
  --metric all --value-labels all \
  --output-dir /path/to/plots/selected
```

未指定长度时，自动为**每一个输入/输出配对分别出图**，不把不同 workload 连成一条曲线。长度参数必须成对使用；指定的 workload 在任一数据集中不存在会报错。默认绘制所有数据集 workload 的并集：某组缺少一个 workload 时，对应位置显示无数据。

`--model` 接受模型原名或对齐后的名称，`--strategy` 精确匹配策略原名，两者均区分大小写、可重复传入。模型筛选分别在各数据集执行；比较多个数据集时，推荐使用基线中的模型名，以便同时选中对齐后的候选模型。策略名以 `matrix_config.json` 的 `name` 为准，不是文件名中的带 TP/PP 后缀标签。

## 输入格式与对齐规则

当前 offhand CSV 的以下字段直接用于解析，**文件可以重命名，名字被截断也不影响识别**：

| CSV 列 | 用途 |
| --- | --- |
| `输入`、`输出`、`并发数` | 正整数 workload 与横轴 |
| `模型`、`并行策略` | 完整模型名和策略名 |
| `TP`、`PP`、`DP` | 正整数并行规模 |
| `EP_ENABLED` | `0/1` 或 `false/true` 开关 |
| `EP_SIZE` | 开启时必须等于 `TP × DP`，关闭时等于 1；缺失时按此规则推导 |
| `状态` | `completed`、`failed`、`incomplete` |
| 指标列 | 所选指标对应的数值，见下表 |

默认样本对齐键为：**对齐后的模型名 + 策略名 + TP/PP/DP/EP + 输入/输出 + 并发数**。不同 DP、EP 开关，以及拓扑相同但名称不同的策略均保留为独立配置，避免把不同调优参数的实验混合。

### 模型名前缀匹配

默认 `--match-model prefix` 以基线数据集中的模型名为准：先精确匹配；没有同名模型时，允许一侧名称等于“前缀 + 分隔符 + 另一侧完整名称”。分隔符支持 `-`、`_` 和空格，匹配区分大小写。例如：

| 基线模型名 | 候选模型名 | 对齐结果 |
| --- | --- | --- |
| `DeepSeek-V4-Flash` | `pcie-DeepSeek-V4-Flash` | 统一为 `DeepSeek-V4-Flash` |
| `nvlink_DeepSeek-V4-Flash` | `DeepSeek-V4-Flash` | 统一为 `nvlink_DeepSeek-V4-Flash` |
| `nvlink-DeepSeek-V4-Flash` | `pcie-DeepSeek-V4-Flash` | 不自动对齐，保留各自名称 |

该规则要求较短的一侧是另一侧的完整后缀，不会任意剥离两侧不同前缀，也不会仅凭共同后缀或模型语义猜测匹配。无法匹配的名称保持原样，跨数据集比较时相应样本可能显示 `missing`。单数据集没有跨环境改名。

模型名发生对齐时，终端会打印 `model match` 映射。一个名称能匹配多个基线模型，或同一数据集的多个不同名称将合并为同一名称时，直接报错；不会静默合并。需要严格保持原模型名时，添加 `--match-model exact`：

```bash
python3 vllm/benchmark_plots/plot_benchmarks.py \
  --dataset A=/path/to/run_a --dataset B=/path/to/run_b \
  --match-model exact --output-dir /path/to/plots/exact
```

### 策略匹配

如果两组数据的策略名称不同，但确认策略本身相同，可显式指定：

```bash
python3 vllm/benchmark_plots/plot_benchmarks.py \
  --dataset A=/path/to/run_a --dataset B=/path/to/run_b \
  --match-strategy topology --output-dir /path/to/plots/topology
```

此时忽略策略名称，以 TP/PP/DP/EP 匹配。`samples.csv` 仍保留原名称；聚合和对比 CSV 的 `strategy` 留空。若因此在单个数据集内产生重复键，默认报错；可用 `--strategy` 先筛选，或在确认可以合并后选择聚合方式。

CSV 支持 UTF-8、UTF-8 BOM 和 GB18030 编码；表头忽略空白，因此 `首 token 时延(ms)` 与 `首token时延(ms)` 均可识别。目录递归扫描默认使用 `*.csv`，可通过 `--glob '*__tp8-*.csv'` 缩小范围；对显式指定的文件不应用 glob。

本工具生成的英文表头结果 CSV 和其他无关表格会被跳过并提示；识别为 benchmark 但缺少必需列、身份信息不完整或维度无效的 CSV 会报错。建议每次传入一个具体 run，避免将整个 `offhand_runs` 下的历史测试无意混在一起。

### 兼容旧版 CSV

没有身份列的旧 CSV，从 `<model>_tp<N>pp<N>[_后缀].csv` 提取模型、TP、PP，默认 DP=1、EP 关闭、策略名为空。输入输出仍读取数据行；若旧文件名声明了 `_4k1k.csv` 等长度，会检查其与行内容一致，`k=1024`。不从当前 offhand 的摘要文件名推断缺失的身份列。

原有 PCIe 文件名前的 `pcie_` 可以通过显式选项去掉，再应用上述模型名匹配规则：

```bash
python3 vllm/benchmark_plots/plot_benchmarks.py \
  --dataset NVLink=/home/daniel/code/prof_data/ds_v4_perf_data \
  --dataset PCIe=/home/daniel/code/prof_data/pcie_ds_v4_perf_data \
  --legacy-strip-prefix PCIe=pcie_ \
  --glob '*_4k1k.csv' \
  --output-dir /tmp/legacy_tpot_compare
```

`--legacy-strip-prefix` 只作用于指定数据集由旧文件名提取的模型名，不改动现代 CSV 的模型列；后续 `--match-model prefix` 对新旧格式均生效。缺少 `状态` 列的数据记为 `legacy`，只按指标有效性判断，不表示已通过当前 offhand 的成功校验。新旧数据对比时，模型名称须能按所选规则对齐，且通常需要 `--match-strategy topology`。

## 指标与图形

| `--metric` | CSV 指标列 | 单位 | 更优方向 |
| --- | --- | --- | --- |
| `tpot`（默认） | `非首token时延(ms)` | ms/token | 更低 |
| `ttft` | `首token时延(ms)` | ms | 更低 |
| `throughput` | `输出吞吐` | token/s | 更高 |
| `per-concurrency` | `单并发输出` | token/s（每个配置并发） | 更高 |
| `duration` | `测试时间` | s | 同等工作量下更低 |
| `all` | 以上所有指标 | 各自单位 | 各自方向 |

`--metric` 可重复使用，每种指标单独出图，不混用单位。TPOT、TTFT 是 offhand 导出的均值；单并发输出沿用 CSV 中“输出吞吐 / 配置并发数”的定义，不等同于 `1000/TPOT`。不从均值推导 P99 或其他原 CSV 未保存的指标。

比值始终为 **候选值 / 基线值**。延迟指标比值小于 1 更好，吞吐指标比值大于 1 更好；绿色表示改善，红色表示退化，接近 1 为中性色。比值热力图、CSV 使用同一份聚合结果。鼠标悬停 SVG 的点或热力单元格可查看数值和状态。

单数据集的并行策略图采用宽幅单列布局，图高随曲线数量增加，图例分为两列。同一 TP/PP/DP 使用相同颜色和点形；EP 开启为实线与实心点，关闭为虚线与空心点。每条有有效数据的曲线右侧显示 `TP/PP/DP · EP` 标签，通过引导线连接最后一个有效点，标签纵向间距至少 23px。拓扑和 EP 均相同但名称不同的策略仍独立绘制，不过样式与末端标签相同，需通过图例和点的悬停信息确认策略名。

多数据集仍按“模型 × 策略”分子图，每条线对应一个数据集，布局由 `--columns` 控制；不使用单数据集的 EP 样式和右侧拓扑标签。

| 图形选项 | 默认值与行为 |
| --- | --- |
| `--linear-x` | 默认 log2 并发轴；指定后改用线性轴 |
| `--y-scale auto/shared/independent` | 默认 `auto`：各子图最大值相近时共享 Y 轴，否则分别缩放 |
| `--value-labels all/last/none` | 默认 `last`：多数据集标注每条线最后一个有效点的数值；单数据集仅显示右侧拓扑标签，数值通过悬停查看。`all` 标注所有有效点数值；`none` 隐藏点数值标签，但单数据集右侧拓扑标签仍保留 |
| `--columns N` | 多数据集每行最多 N 个折线子图，默认 2；单数据集固定单列，忽略此布局设置（参数仍须为正整数） |
| `--title TEXT` | 自定义折线图标题前缀 |

曲线在扫描到的并发网格中遇到缺失或无效样本会断开。完全未出现在任何 CSV 中的未启动 case 无法从 CSV 推断，不会凭空补点。

## 失败、缺失与重复数据

- `failed` / `incomplete` 即使有正数指标，也不会参与曲线、聚合或比值；原始状态和值保留在 `samples.csv`。
- 指标为空、负数、NaN、Inf 或不可解析时视为无效。默认零值也不参与数值分析；`--include-zero` 允许成功/legacy 行的零值，但不会纳入失败行。
- 基线为零时保留绝对值与差值，比值和百分比变化留空，状态为 `zero_baseline`，热力图显示 `ZERO`。
- 缺少对齐样本时状态为 `missing`，热力图显示 `N/A`。失败显示 `FAIL`，无效指标显示 `INVALID`。不会把它们填为 0，全部无有效点时仍能输出诊断图和 CSV。
- 同一数据集出现重复对齐键，默认报错并指出来源。明确要合并重复实验时使用 `--aggregate mean/median/min/max`，仅聚合有效测量值；有被排除的重复样本时汇总状态为 `partial`，同时保留总次数与有效次数。这里的均值是**各次 benchmark 均值的算术平均**，不是按请求数加权的全体请求均值。

例如，专门收集了多次同配置实验的目录可以显式取中位数：

```bash
python3 vllm/benchmark_plots/plot_benchmarks.py \
  /path/to/repeated_runs --aggregate median \
  --output-dir /path/to/plots/repeats
```

状态列若存在但为空或包含未知值，直接报错，避免将未确认的状态作为成功结果。

## 输出文件

例如两组数据、两个 workload，选用 TPOT 时：

```text
<output-dir>/
  tpot_in4096_out1024.svg          # 绝对值折线图
  tpot_in4096_out1024_ratio.svg    # 所有候选相对基线的比值热力图
  tpot_in8192_out2048.svg
  tpot_in8192_out2048_ratio.svg
  samples.csv                    # 筛选后的逐行指标，含失败行、文件路径和行号
  summary.csv                    # 每个对齐键的数值、状态、总次数和有效次数
  comparison.csv                 # 多数据集时生成；每个候选与基线的对齐结果
```

单数据集不生成比值热力图和 `comparison.csv`。三个 CSV 汇总本次选择的全部 workload 和指标，以 `metric` 列区分；通过维度列可关联到图中各个点。

`samples.csv` 是规范化后的指标明细，不是输入 CSV 的逐字副本：`model` 使用对齐后的模型名，不另存 `original_model` 列；无效数值转为空，未选指标及时间段等非分析列不复制。需要核对原始模型名或单元格时，使用 `source` 与 `row_number` 回溯源文件。

`comparison.csv` 中 `ratio = value / baseline_value`，`delta = value - baseline_value`，`change_pct = (ratio - 1) × 100`，百分比方向不随指标改变。例如 TPOT 的 `change_pct=-20` 表示延迟下降 20%；吞吐的 `change_pct=20` 表示吞吐增加 20%。`baseline_status`、`sample_status` 和有效样本次数用于区分缺失、失败或部分成功的聚合。

同名产物会覆盖，输入 CSV 不会被覆盖。程序不会清理输出目录中以前留下的其他文件；切换指标或数据集数量时建议使用新的输出目录。

## 文件与验证

- `plot_benchmarks.py`：统一命令行入口、对齐、聚合、比较及 CSV 导出。
- `benchmark_data.py`：CSV 格式适配、身份/维度校验和指标注册表；新增同类标量指标可扩展 `METRICS`。
- `svg_charts.py`：标准库 SVG 折线图与热力图渲染。
- `test_plot_benchmarks.py`：无需 GPU 的格式、状态处理及端到端回归测试。

```bash
python3 -m unittest discover -s vllm/benchmark_plots -p 'test_*.py' -v
```

如果目前只有 `group.log`，先用现有导出脚本生成 CSV：

```bash
python3 vllm/offhand_batch_test/export_data.py /path/to/group.log \
  --output /path/to/group.csv
python3 vllm/benchmark_plots/plot_benchmarks.py /path/to/group.csv \
  --output-dir /path/to/plots/group
```
