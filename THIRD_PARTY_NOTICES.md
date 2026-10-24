# 第三方来源与许可范围

- vLLM v0.8.3：上游 https://github.com/vllm-project/vllm/tree/v0.8.3 ，Apache-2.0。根 LICENSE 和 upstream/v0.8.3-LICENSE.txt 保留许可文本；覆盖层原始 SPDX 保留。源码重建器校验官方归档 SHA256。
- overlay/vllm 和 overlay/tests：包含上游文件的修改版及新增策略/身份文件；补丁明确相对固定上游的差异。不是维护者从零编写全部推理引擎。
- overlay/agentcache/reference：本项目冻结实验 v1/v2 参考，仅供差分测试；不作为工程运行策略依赖，也不声称完整 ARC。
- Qwen 模型、PyTorch、CUDA 编译扩展及其它运行依赖：不随本仓库分发。使用者应分别核对相应版本的许可、访问和运行条件；本仓库 Apache-2.0 不授予外部资源的额外权利。
- PDF 使用本机中文字体生成文档，没有将字体文件、渲染工具或生成环境打入仓库。

新增整理文件采用根 LICENSE。代码和材料有 AI 辅助，来源与人工验收边界见 README；维护者应确认新增材料分发授权和真实个人贡献。本说明是分发清单，不是对全部权利无争议的保证。
