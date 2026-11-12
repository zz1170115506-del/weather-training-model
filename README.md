# weather-training-model

Task1 / Dataset1 天气四分类。只使用老师提供且保持不变的训练集；无外部数据或测试标签训练。唯一正式入口为 `code/run_task1.py`。真实训练与Kaggle上传由使用者手动执行。

## 当前第七轮：有限配方，快速对照

| 配方 | 主要参数 | 目的 |
| --- | --- | --- |
| HGB强基线 | 1600阶段，lr=.015，15叶，minleaf40，L2=15 | 保留历史最佳普通模型 |
| 自主OvR HGB | 四个二分类头，各1400阶段，lr=.02 | 改变多分类损失分解 |
| 固定OvR/softmax组合 | 各1200阶段，固定50:50概率组合 | 比较不同损失的互补性 |
| 原生类别LightGBM | 1500阶段，15叶，minchild50，L2=20，类别平滑 | 比较不同类别分裂方式 |

配置：[`configs/task1/seventh_run_course_only.json`](configs/task1/seventh_run_course_only.json)。共同五折共20次CV调用，无105次内部重训，无学习融合权重。A/B按开发OOF排序取前两名，可来自同一家族。OOF参与选模，属于开发估计；不冒称独立测试。所有预测由拟合模型生成，无故意改错或分数上限。

第六轮耗时11174秒（约3小时6分）：subag OOF=.917128，普通HGB=.916810，Ordered CatBoost=.912229且其基础CV耗时2550秒。新默认流程不再运行这一慢搜索。历史聚合结果、配置与自主算法保留，未删除训练记录。新轮尚未训练，0.92–0.94是目标，不是已验证成绩。

## 安装与手动运行

使用Python 3.12，在本仓库根目录打开PowerShell；直接调用解释器，不必激活环境。原课程3CSV默认位于仓库外`../Dataset/`，也可显式指定任意目录。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements-round7.txt
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py check --config configs/task1/seventh_run_course_only.json --data-dir ../Dataset
```

已有环境时无需重新创建或重复安装。`check`只检查文件SHA256/schema/依赖，不训练。显示ready后由使用者亲自执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py train --config configs/task1/seventh_run_course_only.json --data-dir ../Dataset
.\.venv\Scripts\python.exe -X utf8 -B code/run_task1.py results
```

每折开始立即显示模型/行数，计算中每30秒显示elapsed；全量重拟合也有心跳。没有虚构完成百分比或ETA。OvR纯模型每阶段共4棵树，固定混合配方共8棵；20次工作流调用不等于20棵树。实际速度需要测量。

输出：`../local_artifacts/task1/<唯一run_id>/`，含冻结配置、源码/数据/折/环境哈希、逐折/分类指标、实际参数与轮数、耗时、选模记录、保存模型和CSV。A/B均是整数1–4、3300行加`id,label`表头；英文`_name.csv`仅供查看，不能上传。每个候选都经过模型重载和CSV哈希复现。

## 老师复验与干净代码包

```powershell
.\.venv\Scripts\python.exe -X utf8 -B code/verify_task1_release.py --output ../task1_release_round7
```

输出目录须不存在；工具只复制runtime白名单、依赖、必要README及第七轮配置，生成SHA256清单，并在隔离目录验证imports、CLI和生成数据的check。它不训练课程数据。包内无数据集、模型、预测、个人文档、虚拟环境、Git目录或历史实验结果。

老师从包根目录安装`code/requirements-round7.txt`，使用自己原课程文件目录传`--data-dir`，按包中README执行。若复验最终采用的提交，使用该次冻结配置、原始数据SHA及代码版本，而不是随意换配方重训。训练完成后的选中配方在`selected_config.json`；该次bundle用于`predict --bundle <路径> --output <新CSV> --data-dir <课程目录>`重放。

## 核心职责与自主技术

- `run_task1.py`：检查、训练、结果、预测、提交文件验证及手工成绩记录。
- `task1/ovr_hgb.py`：自行实现四类二值化、阳性概率对齐、归一化和固定混合；HGB基学习器由sklearn提供。
- `task1/validation.py/features.py/data.py`：共同折、仅折内拟合预处理、原始数据哈希防改动。
- `task1/progress.py`：仅日志心跳，不参与随机种子或模型数值。
- `weather_models.py`、`generative_weather.py`：NumPy手写CART/随机森林、Gini/OOB及直方图混合模型/EM；历史对照的自主技术依据。
- `task1/pipeline.py`：有限配方排序、候选重拟合、模型保存、预测复现；历史嵌套模式只供旧实验重放。
- `configs/task1/`其余配置、`results/`和`outputs/`：明确的历史实验与聚合结果，不会自动运行，不进入干净发布包。

旧的baseline/custom独立脚本已归档到仓库外，避免多个训练入口互相混淆。不要把调用库模型当成全部自主贡献；算法说明、消融和真实效率证据都须在最终报告中解释。仅提高公榜不足以证明私榜或评分满分。

## 测试与协作

```powershell
# 历史可选模型测试需要额外依赖；当前正式训练只需round7依赖。
.\.venv\Scripts\python.exe -m pip install -r code/requirements-boosting.txt
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s code/tests -v
```

测试只用合成数据。共享：核心代码、测试、固定依赖、正式配置、必要README及过滤聚合结果；本地：课程数据、PDF、环境、模型、逐行OOF/预测、完整机器环境、私人MD及停用历史。外部数据配置/来源辅助bundle仍被拒绝。新修改通过新分支与PR由负责人审核，不直接合并main。

方法参考：[OvR](https://scikit-learn.org/1.8/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)、[LightGBM类别支持](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html#categorical-feature-support)。这些方法没有对本课程隐藏测试的成绩保证。
