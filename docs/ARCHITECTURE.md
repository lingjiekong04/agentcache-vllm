# 架构与核心代码阅读

## 所有权边界

BlockPool 持有物理块元数据、引用计数、完整块哈希及 `{hash: {block_id: block}}` 索引。
TwoQueuePolicy 持有冷/热空闲队列、成员映射、有序保护集合和策略累计计数。策略不控制 GPU 张量，不更改引用所有权。

普通块 `ref_cnt=0` 可以仍有缓存。首次重新分配才撤销旧身份；释放不等于清零。
null block 不进入普通空闲队列，不能简单按 ref=0 处理。

## 生命周期推演

两个请求共享 A：引用从 1 到 2，再到 1，最后到 0。只有最后一步能入队。
查哈希不取得所有权；`touch` 先取引用，再晋升。晋升超额可撤销 A 的保护，但 A 若仍有引用，既不进入冷队列，也不能被淘汰。
释放后根据是否受保护入热或冷；分配优先冷，冷尽回退热。复用一个同哈希副本时，只移除该物理 ID。
reset 拒绝发生在改变哈希或策略之前。成功后热区合并入冷区、保护和哈希清理，累计事件计数仍保留。

## 原生传递与限制

EngineArgs -> CacheConfig -> VllmConfig 校验 -> Scheduler -> KVCacheManager -> BlockPool。
参数默认 LRU；Two-Queue 仅限单卡 V1、APC、单 FullAttention 组、纯文本等已声明范围。
支持检查分布在 config 和原生 manager 路径；不是只在演示脚本里约束。

## 指标链

池快照 -> SchedulerStats -> EngineCoreOutputs -> 前端记录 -> Prometheus。
Gauge 表示冷/热空闲、保护数与额度。Counter 对池的绝对累计值求增量：输入 5、5、8 对应导出 5、0、3，避免重复累计。
单次快照读取队列长度/字典长度/计数，不扫描全部块。计数倒退拒绝；reset 不清累计数。
标签沿用 model_name、engine，不引入请求或会话 ID。

## 对应测试

| 不变量 | 代码/检查 |
|---|---|
| 默认 LRU 顺序保持 | check_cpu.py 的官方基线差分 |
| 移植等价性 | 冻结 policy_v2.py 与原生策略有效状态比较 |
| 活动块不可分配 | active/duplicate/reset/capacity 合约 |
| 保护有界、全热回退 | idle_transfer_quota_all_hot_and_metrics |
| 共享取消 | test_agentcache_cancel_shared_prefix_keeps_active_owner |
| 抢占恢复 | test_agentcache_preemption_then_resume_releases_all_blocks |
| 指标增量 | test_prometheus_record_consumes_lifetime_deltas_once |

阅读顺序：策略文件 -> BlockPool -> manager -> 配置链 -> 统计链 -> 两份原生测试。
覆盖层均映射到 SOURCE_MAP.json 的原字节哈希；查看补丁可区分上游与新增代码。
