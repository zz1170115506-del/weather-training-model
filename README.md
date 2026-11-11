# weather-training-model

课程 Dataset1 天气四分类。当前有效方案只使用老师提供的原始 `train.csv`，标签为1 Sunny、2 Cloudy、3 Rainy、4 Snowy，主指标为 Macro-F1。真实训练和 Kaggle 上传由使用者手动执行。

## 当前方案：第六轮，只用课程原始数据

2026-11-11起停用外部来源训练。下载器、外部增强实现、第五轮配置和共享结果已移出当前代码；旧外部配置和来源辅助模型会被拒绝。课程三个CSV保持原始字节不变，第六轮配置固定其SHA256，任一文件变化即在拟合前停止。

第五轮来源辅助结果仅留在仓库外停用归档，不作为当前方案的有效成绩、训练输入或选模依据。当前仅课程数据的历史最佳开发OOF为普通HGB1600的 **0.916810**；它不等于Kaggle分数。0.92–0.94是本轮目标，未经过验证，不是程序对预测的分数限制。

| 第六轮候选 | 方法与主要参数 | 作用 |
| --- | --- | --- |
| r6_hgb_control1600 | HGB1600，lr=.015，15叶，minleaf40，L2=15 | 保留最强固定对照 |
| r6_generative_control700 | HGB700 + 自主直方图混合模型权重，floor=.2 | 保留已有自主加权对照 |
| r6_numpy_forest100 | NumPy手写随机森林100树 | 保留自主算法对照 |
| r6_hgb_subag5 | 自主分层无放回90%采样，5个HGB1200，max_features=.9 | 产生实际多样性并平均概率 |
| r6_hgb_crossfit_weighted | HGB1600 + 内部3折教师置信度温和降权 | 隔离训练样本自身标签对置信度的影响 |
| r6_svc_rbf | 折内标准化，C=4，gamma=.05 | 提供不同于树的平滑决策边界 |
| r6_catboost_ordered_d4 | Ordered，depth4，最多3000轮，one_hot_max_size=2 | 比较不同类别特征处理方式 |

配置为 [`configs/task1/sixth_run_course_only.json`](configs/task1/sixth_run_course_only.json)。所有配方使用相同五折；内部选模三折。子采样集成与普通HGB归为同一家族，避免误称为不同家族。融合只在嵌套开发评估通过既定增益和折胜数门控后额外导出。

## 手动操作

在 **weather-training-model仓库根目录** 的PowerShell终端中执行，不必激活环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r code/requirements.txt -r code/requirements-boosting.txt
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py check --config configs/task1/sixth_run_course_only.json
```

数据应位于仓库外 `../Dataset/`，包含原始 `train.csv`、`test.csv`、`sample_submission.csv`。依赖已经满足时可跳过安装。复现历史环境时使用 `code/requirements-lock.txt`，无需升级已锁定的包。

确认 `status=ready` 后，只有使用者亲自输入以下命令才启动训练：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py train --config configs/task1/sixth_run_course_only.json
```

查看完成结果（不训练）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py results
```

第六轮共有35次基础CV调用，再有105次内部CV调用；子采样成员、置信度教师、CatBoost选轮另外增加计算。日志中的较慢阶段不等于卡死。训练耗时需实际测量。

完成后输出位于 `../local_artifacts/task1/<run_id>/`：冻结配置、代码/数据/折/环境哈希、每折指标及耗时、实际轮数、选模记录、模型和候选CSV。`candidate_a`是开发OOF最好的单模型，`candidate_b`优先为最佳不同家族备选；字母不固定代表加权或普通HGB。未过门控不会生成研究融合提交。

Kaggle上传 `submissions/candidate_a.csv` 或 `candidate_b.csv`，格式为 `id,label`，3300条预测加表头，label必须是整数1–4；不要上传英文 `_name.csv`。

```powershell
$taskRun = Get-Content ..\local_artifacts\task1\latest_training_run.json -Raw -Encoding UTF8 | ConvertFrom-Json
$taskRun.candidates
$taskCandidate = Read-Host '实际上传候选，例如 candidate_a'
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py validate-submission --submission $taskRun.candidates.$taskCandidate
$taskSubmissionId = Read-Host 'Kaggle实际提交编号'
$taskPublicScore = Read-Host '实际Public Macro-F1，英文小数点'
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py record-score --run-dir $taskRun.run_dir --candidate $taskCandidate --submission-id $taskSubmissionId --public-score $taskPublicScore
```

程序不自动提交Kaggle，不读取外部标签，也不故意改错预测以控制分数。最终提交选择须排除已停用的来源辅助候选。

## 方法与文件职责

- `code/run_task1.py`：唯一正式训练/验证/预测入口；`check`与`results`不训练。
- `code/task1/bagged_hgb.py`：自主分层采样、成员种子、类别概率对齐与平均；记录各成员实际轮数、采样哈希和参数。
- `code/task1/crossfit_weights.py`：当前fit部分内部交叉拟合，每行得到一次样本外教师预测；仅高置信冲突行降到(.5,1]，不删除行或更改标签。
- `code/weather_models.py`：NumPy手写CART、Gini切分、bootstrap、特征子采样及OOB随机森林。
- `code/generative_weather.py`：自主直方图混合模型/EM，既有权重控制。
- `code/task1/`其余模块：折内预处理、模型工厂、嵌套选模、融合、工件留档和提交校验。
- `configs/task1/`：共享正式配方；前四轮配置保留，失败实验不抹去。
- `results/task1/`：前四轮经过校验的聚合结果。第六轮尚未由使用者训练，不预填结果。

每个验证折只使用其训练部分拟合预处理、教师及模型；不以外层验证标签早停。分数/耗时/混淆矩阵用于后续报告的真实证据。完整OOF上的调参值不能当作独立测试成绩。纯调用现成模型不足以满足作业自主实现要求，上述自主模块和对照需在最终解释中呈现。

## 复核与协作

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s code/tests -v
```

测试使用合成数据。原始数据、PDF、外部来源、环境、模型、逐条OOF/预测和个人计划不上传。共享代码、正式配置、必要README及过滤后的聚合结果才进入后续新分支/PR。2026-11-11本次迭代仅保存本地版本，没有push或修改线上PR；旧线上来源辅助PR暂不应合并为现行方案。

本轮方法参考：[Bagging/子采样](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.BaggingClassifier.html)、[Confident Learning](https://arxiv.org/abs/1911.00068)、[SVC](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.SVC.html)、[CatBoost参数](https://catboost.ai/docs/en/references/training-parameters/common)。置信度降权只是借鉴样本外置信度原则的自主规则，不是完整复现Confident Learning，也不把难样本当作已知错标。
