# 用户手动上传 GitHub（Windows PowerShell）

此文档不会自动执行。候选目录没有 .git；原工程 Git 保持原样。
目标固定为 https://github.com/lingjiekong04/agentcache-vllm.git 。没有读取登录凭据，也没有验证私有仓库登录状态。

## 1. 在候选目录执行

先在 Windows PowerShell 中进入解压后的 agentcache-vllm 候选目录：

```powershell
$Candidate = (Resolve-Path .).Path
$Publish = Join-Path (Split-Path $Candidate -Parent) 'agentcache-vllm-publish'
if (Test-Path -LiteralPath $Publish) { throw '目标目录已存在，暂停，不覆盖' }
git clone https://github.com/lingjiekong04/agentcache-vllm.git "$Publish"
if ($LASTEXITCODE -ne 0) { throw 'clone 失败，暂停；不要猜测代理或重建原仓库' }
& "$Candidate\tools\stage_upload.ps1" -Destination "$Publish"
```

clone 保留远程现有历史；即使远程为空，也不对原工程 git init。
stage_upload 只复制清单文件，排除 build/reports；目标不干净或已有文件不同即暂停，不自动覆盖、不删除。
如果已有 README/LICENSE 产生冲突，先人工审阅合并，不用 force push 或 reset --hard 解决。

## 2. 审阅后由本人提交、上传

```powershell
git -C "$Publish" status --short
git -C "$Publish" diff --cached --stat
git -C "$Publish" diff --cached --check -- . ':(exclude)overlay/**' ':(exclude)patches/**'
git -C "$Publish" commit -m "Publish evidence-backed AgentCache project"
if ($LASTEXITCODE -ne 0) { throw '提交失败，暂停，先处理身份等实际报错' }
git -C "$Publish" push -u origin HEAD
if ($LASTEXITCODE -ne 0) { throw '上传失败，保留提交，根据报错处理认证或网络' }
```

空白检查只针对新写的交付内容。覆盖层和 unified diff 的原始空白按来源字节保留，不能为消除提示而修改冻结源码或补丁上下文。
不自动填写 Git 邮箱。不把账号密码、token 发给助手，不写入脚本。
若缺提交身份，只在目标 clone 配置本人真实姓名/邮箱；勿未经确认修改全局配置。
若认证或代理失败，先提供去除凭据后的报错；代理端口需本人确认，不能猜测。

## 3. 更新公版清单

README、PDF、文档或工具改变后，必须显式更新 PUBLIC_MANIFEST.json 并重新检查。
历史 accepted-source-manifest.json 不改，核心变更需要新的测试与验收，不允许靠刷新清单冒充通过。
工具见 tools/update_manifest.py；使用 --write 代表明确刷新候选清单，而非历史证据。
