# offhand 自动压测矩阵

`run_matrix.sh` 按 **模型 → 并行策略 → 输入输出组合 → 并发** 顺序测试。每个“模型 × 并行策略”只通过 `start.sh` 启动一次 vLLM；服务就绪后，通过一次 `batch.sh` 执行该组全部 case，随后停止服务、等待日志写完，调用 `export_data.py` 导出本组 CSV，完成归档后再启动下一组。

## 快速使用

先修改 [matrix_config.json](matrix_config.json)，然后在已配置好 vLLM 的环境目录执行：

```bash
# 路径替换为脚本所在位置。保留当前工作目录，供 uv 查找项目环境。
SCRIPT=/home/daniel/code/prof_data/scripts/offhand/run_matrix.sh
CONFIG=/home/daniel/code/prof_data/scripts/offhand/matrix_config.json

# 校验配置并显示展开后的矩阵，不启动服务、不创建结果目录。
bash "$SCRIPT" dry-run --config "$CONFIG"

# 默认 start；内部使用 nohup，并创建独立 session，命令返回后可以关闭终端。
bash "$SCRIPT" start --config "$CONFIG"
```

启动命令输出本次 `Run directory`、调度 PID 和 `runner.log` 路径。每次运行创建新的目录，配置会快照到 `config.json`；修改原配置不影响已经启动的任务。

```bash
# 替换为启动命令输出的实际目录。
RUN=/path/to/offhand_runs/20260915-120000-000000-12345

tail -f "$RUN/runner.log"
bash "$SCRIPT" status --run-dir "$RUN"
bash "$SCRIPT" stop --run-dir "$RUN"
```

`stop` 发送停止请求后返回；清理和已有 case 的归档结束后，`status.json` 的 `state` 变为 `stopped`。退出 `tail` 不影响任务。前台调试可用 `bash "$SCRIPT" run --config "$CONFIG"`；此模式的调度消息输出到终端，仍会保存各组日志、配置和状态。

## 配置说明

当前示例配置：DeepSeek-V4-Flash，TP8、TP4/PP2、PP8 三个策略，每组执行 4096 输入 / 1024 输出和 8 个并发，共 24 个 case。

| 字段 | 含义 |
| --- | --- |
| `models` | 模型数组；`name` 是日志标签及服务模型别名，`path` 是实际模型路径或 Hugging Face ID。压测 tokenizer 使用 `path` |
| `strategies` | 策略数组；每项配置唯一 `name`、`tp`、`pp`、`dp`、布尔值 `ep`。`ep: true` 添加 `--enable-expert-parallel` |
| `workloads` | 输入输出配对，如 `[[4096,1024],[8192,2048]]`；每一对都会遍历全部并发 |
| `concurrencies` | 如 `[1,4,16,32,64,128,256]` |
| `serve_args` | 额外 vLLM 启动参数，使用 JSON 字符串数组，参数和值分开写 |
| `env` | 环境变量对象，值必须为字符串，如 `{"CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5,6,7"}` |
| `output_dir` | 输出根目录，默认 `./offhand_runs` |
| `work_dir` | 可选，`uv` 命令的工作目录，默认是启动调度命令时的目录 |
| `continue_on_error` | 某一组失败并完成清理后，是否继续下一组，默认 `true`；整体状态仍然是 `failed` |
| `lock_file` | 可选，默认 `/tmp/offhand-vllm-<用户UID>.lock`，防止同一用户同时运行两个矩阵 |

`serve_args` 和 `env` 可以同时出现在配置顶层、模型项、策略项。参数按“公共 → 模型 → 策略”顺序传给 `start.sh`；环境变量同名时后者覆盖前者。模型、端口、TP/PP/DP/EP 等受调度管理的参数应使用对应字段，不能在 `serve_args` 中重复指定。

增加模型时复制 `models` 中的一项，修改 `name` 和 `path`，并设置适合该模型的 `serve_args`。示例将 `deepseek_v4` 的 tokenizer / parser 参数放在模型项中；公共参数沿用原 `start.sh` 的调优设置，更换模型时也应按实际支持情况调整。

策略还可以设置专属环境和参数，例如：

```json
{
  "name": "tp4_pp2_ep",
  "tp": 4,
  "pp": 2,
  "dp": 1,
  "ep": true,
  "serve_args": ["--gpu-memory-utilization", "0.85"],
  "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
}
```

`start.sh` 默认保留原脚本清除 NCCL / all-reduce 环境变量的行为。如果要在 `env` 中配置 `NCCL_P2P_DISABLE` 等变量，同时加入 `"RESET_COMM_ENV": "0"`。

所有相对的配置路径（包括 `output_dir`、`work_dir`、`lock_file`）相对于**启动命令的工作目录**。模型路径直接传给 vLLM，相对模型路径相对于 `work_dir`；建议使用绝对路径。策略名称与模型名称在各自数组内必须唯一，输入输出配对和并发不能重复。

### 服务等待与切换

| `server` 字段 | 默认值 | 行为 |
| --- | --- | --- |
| `host` / `port` | `0.0.0.0` / `8000` | 本机绑定地址；通配地址通过本机回环地址探测 |
| `startup_timeout` | 1800 秒 | 加载模型、初始化引擎及健康检查的等待上限 |
| `health_interval` / `health_timeout` | 2 / 5 秒 | 探测间隔 / 每次 HTTP 请求超时 |
| `ready_successes` | 2 | 连续通过 `/health` 和 `/v1/models` 中模型别名校验后才开始压测 |
| `health_failure_threshold` | 3 | 压测期间连续健康检查失败达到此次数，终止当前组 |
| `shutdown_timeout` | 120 秒 | 给每个受管进程 session 的 TERM 清理时间，超时后 KILL |
| `settle_seconds` | 5 秒 | 清理完成后的额外等待时间 |

启动前检查端口是否空闲；已有服务占用时直接报错。清理仅针对本次启动的服务和压测 session，包含 session 内另建进程组的 worker。只有确认这些进程退出且端口释放后才进入下一组；清理失败会终止整个矩阵，即使 `continue_on_error` 为 `true`。

### 压测参数

| `benchmark` 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `request_rate` | `"inf"` | 请求发送速率 |
| `prompts_multiplier` | 1 | 每个 case 的请求数 = 并发 × 此倍数 |
| `case_timeout` | 3600 秒 | 单个 case 的超时；0 表示不限时 |
| `continue_on_error` | `true` | 普通 case 失败后，是否继续同组剩余 case；整组仍返回失败 |

成功必须同时满足：压测命令退出码为 0、结果 JSON 存在且可解析、`completed` 等于请求数、存在的 `failed` 字段为 0，以及日志写入成功。服务异常或 case 超时会中断该组，完成清理后按顶层 `continue_on_error` 决定是否测试下一组。

每个 case 的压测进程及其子进程都有独立进程组。case 超时后先发送 TERM，最多等待 30 秒，再强制 KILL 遗留进程，因此超时后的清理可能额外耗时约 35 秒。

## 日志与结果

```text
offhand_runs/<本次运行时间与PID>/
  config.json                    # 实际使用的配置快照，含默认值
  runner.log                     # 后台模式的调度进度与异常
  status.json                    # running / completed / failed / stopped
  summary.json                   # 每组状态、错误、起止时间、目录
  001-DeepSeek-V4-Flash__tp8-tp8-pp1-dp1-epoff/
    group.log                    # 整组大日志：服务 + 所有case + 清理过程
    <模型>__<策略与TP-PP-DP-EP>__<输入输出组合>__<并发列表>__<运行标识>.csv
    server.log                   # 单独的服务输出
    status.json                  # 本组结果
    cases/
      DeepSeek-V4-Flash-tp8-tp8-pp1-dp1-epoff-in4096-out1024-c16-<运行标识>.log
    results/
      DeepSeek-V4-Flash-tp8-tp8-pp1-dp1-epoff-in4096-out1024-c16-<运行标识>.json
```

`group.log` 中 `[server]` 和 `[bench]` 标识来源；包含单 case 的参数、完整命令、输出及结果判定。case 文件名中的模型与策略名称会清理路径分隔符等字符，并各截取前 64 字符。

配置中的 `env` 按“公共 → 模型 → 策略”合并后，启动日志会在命令前以 JSON 记录这些变量的实际值。调度日志中的 `server environment` / `bench environment` 表示传给对应脚本的环境；`server.log` 中的 `Server environment` 表示 `start.sh` 清理通信变量后的环境，每个 case 日志中的 `Environment` 表示压测环境，这些输出也会汇总到 `group.log`。已清除的变量显示为 `null`，空字符串保留为 `""`；仅记录配置中指定的变量，值中的换行等字符会转义。

### 每组自动归档 CSV

调度器在服务和压测进程清理完成、日志输出全部写入后，同步执行：

```bash
python3 /path/to/offhand/export_data.py "$RUN/<组名>/group.log" \
  --output "$RUN/<组名>/<包含本组配置信息的文件名>.csv"
```

CSV 文件名包含模型、策略、TP/PP/DP/EP、输入输出配对、并发列表以及运行标识和组序号，单独复制到交付目录也可以识别来源。例如：

```text
DeepSeek-V4-Flash__tp8-tp8-pp1-dp1-epoff__in4096-out1024__c1-4-8-16-32-64-128-256__20260915-120000-000000-12345-g001.csv
```

多个输入输出配对使用 `+` 连接，例如 `in4096-out1024+in8192-out2048`；并发列表如 `c1-4-16`。自动归档的文件名描述本组配置范围，实际已启动或完成的 case 以 CSV 数据行及其状态为准。

名称或矩阵过长时，文件名会保留可读摘要并附加摘要码，限制在 240 字节内；CSV 内容中保留完整模型名、策略和每行输入输出、并发。运行标识和组序号用于区分不同次运行和不同组，避免单独收集 CSV 时重名。实际文件路径也记录在本组 `status.json` 的 `csv_file` 中。

导出期间总任务的 `phase` 为 `exporting`。导出命令及完成消息可在 `runner.log` 查看；脚本输出以 `[export]` 前缀写入 `group.log`。调度器会检查导出退出码、CSV 是否可读，以及数据行数是否等于本组 case 日志数，检查结束后才进入下一组。

CSV 保留原 `export_data.py` 的前 9 列：**输入、输出、并发数、单并发输出、输出吞吐、首 token 时延(ms)、非首 token 时延(ms)、测试时间、测试时间段**，按输入、输出、并发排序。其中单并发输出为输出吞吐除以配置并发数，TTFT / TPOT 使用均值。

当前矩阵日志还会追加 **状态、用例、模型、并行策略、TP、PP、DP、EP_ENABLED、EP_SIZE** 列。模型和策略在 CSV 中保留完整原文，因此文件重命名或单独交付后仍能识别每条数据的配置。`completed` 表示 case 成功，`failed` 表示 case 明确失败，`incomplete` 表示中断等原因导致缺少最终结果标记。缺失指标留空；失败和中断的已启动 case 也会保留一行。尚未启动的 case 不生成数据行。服务启动失败且没有 case 日志时跳过导出。

每组 `status.json` 及总 `summary.json` 会记录 `export_state`（`completed` / `failed` / `skipped`）、导出目标 `csv_file`，成功时还有 `csv_rows`，实际正常退出时记录 `export_exit_code`；失败时记录 `export_error`。导出失败会使原本成功的组变为失败，是否继续下一组由顶层 `continue_on_error` 控制。单次导出最多等待 300 秒，超时后清理导出进程。

已有日志也可以手动补导出。单参数调用在日志旁输出 CSV：识别到完整模型及并行信息的矩阵日志，会根据实际数据行生成包含配置的文件名；旧版或缺少身份信息的日志仍使用日志同名 CSV。也可以通过 `--output` 指定文件名。

```bash
python3 /path/to/offhand/export_data.py /path/to/group.log
```

导出脚本兼容旧版无前缀日志和当前 `[bench]` 日志，忽略服务输出，使用 UTF-8 BOM 编码以方便 Excel 打开；文件写完后才替换目标 CSV。旧版日志保持原来的 9 列。

## 单独使用原脚本

`start.sh` 默认仍用 nohup 启动服务并跟随日志；`--no-tail` 仅启动，`--foreground` 由调度器使用。默认参数保持原 DeepSeek 配置。

```bash
MODEL=/models/DeepSeek-V4-Flash/main MODEL_NAME=DeepSeek-V4-Flash \
TP=4 PP=2 DP=1 EP=false LOG=./flash_tp4pp2.log \
bash /path/to/offhand/start.sh --no-tail

MODEL=DeepSeek-V4-Flash TOKENIZER=/models/DeepSeek-V4-Flash/main \
MODEL_NAME=DeepSeek-V4-Flash STRATEGY_NAME=tp4_pp2 TP=4 PP=2 DP=1 EP=false \
WORKLOADS_JSON='[[4096,1024],[8192,2048]]' CONCURRENCIES_JSON='[1,4,16]' \
GROUP_LOG=./flash_tp4pp2_all.log \
bash /path/to/offhand/batch.sh
```

## FAQ

### 为什么每个 case 都要为 `vllm bench serve` 单独生成 `--seed`？

主要是为了减少服务端未关闭 prefix caching 时，重复输入命中前缀缓存而导致吞吐虚高的影响。`vllm bench serve` 的默认 seed 为 `0`；当前脚本使用 `--dataset-name random`，固定 seed 会使随机数据可复现，因此对同一服务反复压测时，可能复用之前留在 prefix cache 中的 prompt。官方文档将这种影响称为 “inflate throughput”。

`batch.sh` 在每个 case 开始时，通过 `secrets.randbits(32)` 生成一个 32 位随机整数并传给 `--seed`，以降低不同 case 之间因固定 seed 重复生成输入、复用缓存的风险。更换 seed 不等于关闭 prefix caching，也不能保证完全没有缓存命中；同一次压测中的共享前缀仍可能命中缓存。

参考 [vLLM v0.29.0 Benchmark CLI 的 prefix cache 提醒](https://github.com/vllm-project/vllm/blob/v0.29.0/docs/benchmarking/cli.md#L104-L111)。链接固定到编写本条 FAQ 时的最新 release（2026-09-16 核对），避免 `main` 后续更新导致对应说明变化或移除。

## 依赖与验证

本压测流程已在 **vLLM 0.29.0** 上测试通过。

运行环境：Linux（使用 `/proc` 跟踪进程）、Python 3.9+ 标准库、Bash 4+、`nohup` / `tee` 等 GNU 工具，以及已安装相应 vLLM 的 `uv` 环境。调度器只负责本机进程生命周期；远程节点、外部 Ray 集群、模型显存容量和具体模型对策略的支持需由实际部署提供。

在仓库根目录执行无需 GPU 的模拟集成测试（会监听临时本机端口）：

```bash
python3 -m unittest discover -s scripts/offhand -p 'test_*.py' -v
```

真实模型加载与吞吐性能需在目标 GPU 机器上验证。

EP 标注：配置中的 `ep` 和原始日志中的 `EP=true/false` 始终表示开关。CSV 的 `EP_ENABLED` 用 1/0 表示开关，`EP_SIZE` 在开启时为 `TP × DP`（不乘 PP），关闭时为 1。文件名、组目录及 case 名使用 `ep8` 等实际规模，关闭时使用 `epoff`。旧日志中的布尔 EP 仍可直接导出。
