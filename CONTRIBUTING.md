# Dataset1 协作约定

本仓库用于双方协作完成 Task1。日常以 GitHub PR 审阅和合并；不要把本地环境、课程材料和比赛原始数据上传。

## 每次工作流程

以下命令从仓库根目录执行。开始前先检查工作区；有未提交改动时先明确归属并保存，不能用 `reset --hard` 或强制覆盖解决同步问题。

```bash
git status --short --branch
git switch main
git pull --ff-only
git switch -c docs/short-description
```

分支按目的命名，例如 `feat/task1-cv`、`experiment/task1-cleaning`、`fix/task1-submission`、`docs/task1-development-plan`。一个分支集中完成一件可审阅的事情。

完成代码/实验或文档后：

```bash
git diff --check
git diff
git add <explicit-file-paths>
git diff --cached
git commit -m "Describe the concrete change"
git push -u origin <branch-name>
```

在 GitHub 创建从该分支到 `main` 的 Pull Request，填写变化、依据和验证，邀请仓库负责人 review；负责人认可后再合并。PR 模板在 `.github/pull_request_template.md`。不要自动合并或强推覆盖协作者改动。

并行工作期间若 `main` 更新，先 `git fetch origin`，检查差异后将 `origin/main` 合并到工作分支并解决冲突、重跑受影响验证；已共享分支优先用 merge，避免改写历史。合并完成后本地回到 main 并 `pull --ff-only`，再开始下一件工作。

首次提交须使用本人的 Git 署名；只需配置当前仓库，无须改变全局设置。Push/PR 另外需要有仓库权限的 GitHub 登录；不要把令牌写进文件、远程 URL 或提交。没有权限时保留本地提交，由有权限的成员推送或采用 fork PR，并准确记录状态。

## 应该提交什么

| 入库 | 留本地 |
| --- | --- |
| 源码、必要测试、固定依赖、相对路径配置 | `.venv`、本地 IDE 设置、缓存、个人路径 |
| 方法/开发计划、README、实验汇总 | 原始训练/测试 CSV、课程 PDF、下载包 |
| 报告所用图、聚合指标、脱敏硬件摘要 | 模型二进制、逐条 OOF/预测、大日志 |
| 提交编号、配置/数据/预测文件哈希、复现命令 | Kaggle/GitHub 凭据、`.env` 等秘密 |

提交前逐项检查 staged diff，不使用 `git add .` 盲目纳入实验目录。新结果目录默认被忽略时，只定向允许需要的汇总和图，不能把整个 outputs 目录开放。

正式课程压缩包可包含生成所需的额外工件，按课程规则准备；这不代表公共 GitHub 应存放所有数据或大工件。

## 实验 PR 的最小信息

- 数据 SHA256、代码 commit、配置和随机种子/折规则。
- 相同 folds 的基线与变化后结果，Macro-F1 的均值/标准差和 OOF 统计量分别列出。
- 各类表现及必要消融；若使用融合，说明权重学习与外层评估隔离。
- CPU/线程/依赖及计时口径；说明是否包含预处理。
- 精确运行命令、实际执行的验证及已知局限。

文档 PR 无需训练模型；代码 PR 运行与改动对应的正确性或流程检查。不要把未执行的检查写成通过。两位成员互相复跑关键实验，报告贡献比例依据实际工作填写。

开发目标见 [Task1 开发计划](docs/task1-development-plan.md)，最终验收见 [评分与交付检查表](docs/task1-delivery-checklist.md)。
