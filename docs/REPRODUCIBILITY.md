# 复现分层与 Linux 双终端运行

## 0. 先区分三种“可复现”

1. **公版源码/聚合完整性**：Python 3.11+ 标准库即可，不需要模型或 GPU。
2. **抽取定义检查**：还需要固定官方基线包；不导入完整 vLLM/Torch，不能称为原生 pytest。
3. **真实服务复现**：需要兼容 Linux 环境、同版本扩展、模型、GPU 和隔离测试工具；新机器尚未重新验收。

本项目没有训练/微调步骤。不提供自动下载依赖或模型的脚本，不升级驱动，不改原 site-packages。

## 1. 本地 CPU 检查

在 Windows PowerShell 或 Linux 项目根目录执行 README 的标准库命令。
已有基线包可直接传入：

```text
python -S -B tools/run_checks.py --archive /path/to/vllm-v0.8.3.tar.gz --output reports/cpu-new
```

上面的 /path 是使用者选择的实际基线位置，Windows 改用自己的完整盘符路径。
重复输出拒绝覆盖。失败日志留在 reports，不删除或用同名覆盖重试。

工具会重建 build/vllm-0.8.3；Windows 支持上游超长文件名。部分失败目录必须保留并检查，不能通过改基线 SHA 绕过。
已存在且有效的重建树会复查记录中的全部文件哈希，再运行测试。

## 2. 原生环境前置条件（Linux）

历史环境：Python 3.11.16、vLLM 0.8.3、A100-SXM4-80GB、驱动 550.54.14，CUDA 扩展来自原同版本安装。
历史运行记录没有完整锁定所有 wheel 及模型分片，所以不声称任意新环境字节级复现。
上游依赖声明见重建树 requirements/ 和 pyproject.toml；不要把它们当完整的已验收 wheel 锁。

在 Linux **公版仓库根目录**，用自己的实际路径替换以下变量：

```bash
export AGENTCACHE_PYTHON=/absolute/path/to/existing-v083-env/bin/python
export AGENTCACHE_HF_HOME=/absolute/path/to/existing/huggingface-cache
source tools/env.sh
repo="$PWD"
cd "$AGENTCACHE_PROJECT"
"$AGENTCACHE_PYTHON" -u agentcache/prepare_runtime.py
"$AGENTCACHE_PYTHON" -u agentcache/prepare_runtime.py --verify
```

只有新源码树会接收复制的 .so；原环境不修改。缺扩展、版本/核心哈希不匹配、目标已存在时暂停。不要直接执行 pip install -e . 改写原环境。
公版运行准备清单是新生成的；历史被验收清单另存在 evidence，二者不能混称。

## 3. Linux 原生 CPU / pytest（不启动模型）

先确认隔离 pytest 已存在，并有上游测试所需的离线模型配置缓存；缺失时停止，不下载或跳过测试伪造通过。
在重建后的源码根目录执行，输出标签必须新建：

```bash
mkdir -p agentcache/results
"$AGENTCACHE_PYTHON" -u agentcache/check_cpu.py \
  --archive /absolute/path/to/vllm-v0.8.3.tar.gz \
  --reference agentcache/reference \
  --output agentcache/results/native-cpu-new.json

"$AGENTCACHE_PYTHON" -u agentcache/check_spawn_v2.py \
  --policy lru --timeout 600 --output agentcache/results/spawn-lru-new.json
"$AGENTCACHE_PYTHON" -u agentcache/check_spawn_v2.py \
  --policy two_queue --timeout 600 --output agentcache/results/spawn-two-queue-new.json
```

不要执行旧 check_spawn.py 的短等待版本。当前公版只在本地执行 spawn 的模拟合约，没有重新运行 Linux spawn。

原生 pytest 命令（在 **Linux 重建源码根目录**；先替换隔离 pytest 路径）：

```bash
test_deps=/absolute/path/to/isolated/pytest-directory
test ! -e agentcache/results/native-pytest-new.xml && \
env PYTHONPATH="$test_deps:$AGENTCACHE_PROJECT" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$AGENTCACHE_PYTHON" -u -m pytest \
  --confcutdir=tests/v1/core tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_scheduler.py -v --maxfail=1 \
  --junitxml=agentcache/results/native-pytest-new.xml
```

不要把本地抽取测试成功写成这条 Linux 命令已执行。

## 4. 固化输入（Linux 重建源码根目录）

缓存 tokenizer 必须存在，工具使用离线模式。只在不存在的输出名生成：

```bash
mkdir -p agentcache/results
"$AGENTCACHE_PYTHON" -u agentcache/demo.py build --profile quick \
  --output agentcache/results/inputs-quick-new.json
"$AGENTCACHE_PYTHON" -u agentcache/demo.py build --profile demo \
  --output agentcache/results/inputs-demo-new.json
```

quick 是 0.5B，demo 是 7B。一个配对必须使用同一份输入，不因输出差异修改输入/阈值。
7B 固定 revision 见 README；0.5B 历史 revision 未完整记录。新复现应自行记录实际模型来源/分片哈希，不能补写成历史事实。

## 5. 启动前检查（Linux 终端 B）

每次使用同一块物理 GPU 0，单策略单场景全新服务。GPU 必须空闲，未知进程不杀：

```bash
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used --format=csv
curl -sS --max-time 3 -o /dev/null -w 'health_http=%{http_code}\n' \
  http://127.0.0.1:8002/health
```

0 MiB 和 health 000 是预期前置状态，但连接拒绝不单独证明端口可以 bind。已有服务到原终端正常 Ctrl+C 后等待释放。

## 6. 最小 7B 抗扫描演示（Linux 终端 A）

每个终端先在公版仓库设置第 2 节环境，再进入重建源码根目录。以下为 LRU：

```bash
set -o pipefail
export AGENTCACHE_RUN_ID=demo-scan-lru-new
log=agentcache/results/demo-scan-lru-new.log
result=agentcache/results/demo-scan-lru-new.json
if [ -e "$log" ] || [ -e "$result" ]; then
  echo '已有输出，暂停，不覆盖'
else
  "$AGENTCACHE_PYTHON" -u -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-7B-Instruct \
    --host 127.0.0.1 --port 8002 --dtype bfloat16 --max-model-len 8192 \
    --num-gpu-blocks-override 1024 --block-size 16 --gpu-memory-utilization 0.2 \
    --max-num-seqs 1 --enable-prefix-caching --seed 0 --generation-config vllm \
    --disable-log-requests --gpu-eviction-policy lru --gpu-cache-protected-ratio 0.5 \
    2>&1 | tee "$log"
fi
```

等待 Application startup complete，A 保持运行。检查 AGENTCACHE_NATIVE_ENGINE 的实际策略/源码身份，而不是只看命令参数。

## 7. 采集（Linux 终端 B，重建源码根目录）

```bash
curl -fsS --max-time 5 http://127.0.0.1:8002/health && \
"$AGENTCACHE_PYTHON" -u agentcache/demo.py run \
  --policy lru --scenario scan_return \
  --manifest agentcache/results/inputs-demo-new.json \
  --server-log agentcache/results/demo-scan-lru-new.log \
  --output agentcache/results/demo-scan-lru-new.json
```

成功应为 complete=true requests=75。失败/中断保留部分 JSON 与日志，不自动重试。
成功后 A 正常 Ctrl+C，等待 GPU 0 释放和端口停止，再从全新服务运行 Two-Queue。
第二次将策略改为 two_queue，run_id/log/output 中 lru 改为 two-queue；其它配置及 manifest 保持一致。不要改成实验名 two_queue_v2。

```bash
"$AGENTCACHE_PYTHON" -u agentcache/demo.py compare \
  --lru agentcache/results/demo-scan-lru-new.json \
  --two-queue agentcache/results/demo-scan-two-queue-new.json \
  --output agentcache/results/demo-scan-comparison-new.json
```

有输出差异则保存并审查，先区分迁移错误与既有 APC 复用/重算路径差异，不自动忽略或改输入。

## 8. 其它场景和生命周期

其它场景分别使用 agent_sessions、working_set_shift、cold_only，每次新启动、独立输出名。quick 使用 0.5B、2048 context、256 blocks，其他固定配置一致。

生命周期需要 **全新 max_num_seqs=4 服务**，不要沿用串行服务：

```bash
"$AGENTCACHE_PYTHON" -u agentcache/lifecycle.py --policy lru \
  --manifest agentcache/results/inputs-demo-new.json \
  --server-log agentcache/results/demo-lifecycle-lru-new.log \
  --output agentcache/results/demo-lifecycle-lru-new.json
```

Two-Queue 对应改策略和独立文件名。必须实际观察抢占以及 running/waiting=0、空闲块恢复；没有发生压力不能宣称通过。
本页是用户手动复现命令，本次发布整理未运行这些 GPU 步骤。
