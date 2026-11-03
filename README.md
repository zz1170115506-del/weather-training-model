# weather-training-model

数据集 1 的天气四分类实验项目。目前已完成数据检查、基线比较、NumPy 手写 CART 决策树与随机森林，以及测试集预测生成。交叉验证和消融实验留待后续完成。

Task1 的后续工作见 [详细开发计划](docs/task1-development-plan.md) 和 [评分与交付检查表](docs/task1-delivery-checklist.md)；分支、PR、数据与实验结果的协作规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。计划中列出的新模块尚未实现，当前可运行入口如下。

## 当前结果

训练集共 9,900 条记录，四类各 2,475 条；测试集共 3,300 条。包含 7 个数值属性和 3 个类别属性。未发现缺失值、训练集重复特征记录或训练与测试之间的相同特征记录。

固定随机种子 `5434`，采用一次分层 80% / 20% 划分，训练 7,920 条、验证 1,980 条。所有模型使用相同划分。

| 模型 | 验证集 Macro-F1 |
| --- | ---: |
| 多数类参照 | 0.1000 |
| 逻辑回归 | 0.8698 |
| 决策树 | 0.9079 |
| 随机森林 | 0.9125 |
| 直方图梯度提升树 | **0.9153** |

这些分数用于初步模型选择，尚未经过独立最终测试，也不是 Kaggle 成绩。完整参数、各类指标和混淆矩阵保存在 `outputs/dataset1_baseline/details.json`，计时结果保存在 `metrics.csv`。运行时间随硬件和环境变化。

![基线比较与混淆矩阵](outputs/dataset1_baseline/baseline_comparison.png)

## 自主实现结果

复用相同的分层 80% / 20% 划分和预处理，比较手写模型与参数一致的库模型。单树深度上限为 8、叶节点最少 5 条；森林使用 100 棵树、叶节点最少 2 条、每个节点随机考虑约 sqrt(p) 个编码后特征。

| 模型 | 验证集 Macro-F1 |
| --- | ---: |
| NumPy CART | 0.9079 |
| scikit-learn CART | 0.9079 |
| NumPy 随机森林（100 棵） | 0.9134 |
| scikit-learn 随机森林（100 棵） | 0.9159 |

此处两种森林均为 100 棵树；上面的早期基线森林为 300 棵树。完整实验记录见 `outputs/dataset1_custom/`，这些仍是单次验证划分的初步结果。

![手写模型比较](outputs/dataset1_custom/custom_comparison.png)

`code/weather_models.py` 的训练和预测只依赖 NumPy，自己实现了：

- 排序候选特征，用累计类别计数计算各切分点的加权 Gini 不纯度，选择最优二叉切分。
- 限制最大深度、最少叶节点样本和相对整棵树的最小不纯度下降；保留零增益切分以学习 XOR 等深层交互。
- 用叶节点的类别比例输出概率，批量遍历树进行预测。
- 对训练记录有放回抽样，在每个节点随机选择候选特征，平均各树的类别概率。
- 在抽样遗漏某类时保持全局类别顺序，避免概率列错位。
- 只使用未抽到某条记录的树计算该记录的袋外（OOB）预测。
- 按不纯度下降计算特征重要性，并汇总独热编码列对应的原始属性。

Bootstrap 重复记录按多次观测计数，叶节点样本下限也采用这个口径。随机数、并列切分的处理和抽样记账与 scikit-learn 存在实现差异，因此参数一致并不意味着树结构或预测完全相同。预处理、指标和对照模型仍使用 scikit-learn。

数学定义与算法参考：[决策树](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)、[随机森林](https://scikit-learn.org/stable/modules/ensemble.html#forest)。

## 获取数据

通过课程指定的 Kaggle 比赛入口获取数据，在项目根目录建立以下目录：

```text
comp-5434-20261-project-task-1/
    train.csv
    test.csv
    sample_submission.csv
```

原始比赛数据和课程 PDF 不包含在仓库中。逐条记录的划分、验证预测、测试预测和概率文件在本地运行时生成，并由 `.gitignore` 排除。

## 运行

建议使用 Python 3.12。以下命令适用于 macOS / Linux，在仓库根目录运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/dataset1_baseline.py
.venv/bin/python code/dataset1_custom.py
.venv/bin/python -m unittest discover -s code/tests -v
```

Windows PowerShell 示例：如果数据位于本仓库外层的 `Dataset` 文件夹，从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements.txt
.\.venv\Scripts\python.exe code/dataset1_baseline.py --data-dir ..\Dataset --output-dir outputs/local_baseline
.\.venv\Scripts\python.exe code/dataset1_custom.py --data-dir ..\Dataset --output-dir outputs/local_custom
.\.venv\Scripts\python.exe -m unittest discover -s code/tests -v
```

命令直接使用虚拟环境的 Python，不要求 PowerShell 激活脚本。`outputs/local_*` 保持本地，不会覆盖仓库中的参考结果；实际数据位置不同则修改 `--data-dir`。

指定其他数据位置或输出位置：

```bash
.venv/bin/python code/dataset1_baseline.py \
  --data-dir /path/to/dataset1 \
  --output-dir outputs/local_run
```

更多参数和输出说明见 `code/README.txt`，或运行：

```bash
.venv/bin/python code/dataset1_baseline.py --help
```

## 数据处理与验证

- `id` 只用于记录匹配，不作为模型特征。
- 缺失值填补、数值标准化和类别独热编码仅在当前训练部分拟合。
- 约 3% 的湿度或降水百分比记录超过 100；当前保留原值，清洗策略留待对比实验验证。
- 检查训练与测试字段、ID 唯一性、样例 ID、数值有效性和重复特征记录。发现训练集重复特征时停止，提示采用分组验证。
- 所有模型使用同一验证集，记录训练集与验证集 Macro-F1、各类指标、混淆矩阵及训练和预测时间。
- 以验证集 Macro-F1 最高的基线在完整训练集上重新训练，生成本地测试预测。
- 已核对原始文件哈希、训练和验证记录互不交叉、指标重算、概率和、预测行数与 ID 顺序；两次运行的划分、预测、概率和分数完全一致。计时不要求相同。

## 标签格式待核对

训练集标签为整数：`1 = Sunny`、`2 = Cloudy`、`3 = Rainy`、`4 = Snowy`。当前下载的提交样例中，`label` 填写的是英文 `Rainy`。

脚本生成 `baseline_predictions_integer.csv`，使用与训练集一致的整数标签。实际 Kaggle 提交编码仍需核对。本脚本不会自动提交比赛。

## 文件结构

```text
code/
    dataset1_baseline.py       数据检查、模型比较与预测生成
    dataset1_custom.py         手写模型与参数一致的库模型对照
    weather_models.py          只依赖 NumPy 的 CART 与随机森林
    tests/                     最优切分、边界、袋外预测和复现检查
    requirements.txt          固定版本的主要依赖
    README.txt                运行细节与本地输出说明
outputs/dataset1_baseline/
    audit.json                数据概况、检查结果和源文件哈希
    details.json              模型参数、各类分数及混淆矩阵
    metrics.csv               基线分数和计时
    data_overview.png          类别分布与百分比异常统计
    baseline_comparison.png    基线比较及最佳模型混淆矩阵
outputs/dataset1_custom/
    audit.json                数据检查与源文件哈希
    details.json              模型参数、结构、OOB 指标和计时
    metrics.csv               参数一致的手写模型与库模型分数
    feature_importances.csv   编码后特征的不纯度重要性
    original_feature_importances.csv  汇总至原始属性的重要性
    custom_comparison.png     分数对比和手写森林混淆矩阵
    feature_importance.png    手写森林的原始属性重要性
```

运行时还会生成本地环境信息、记录划分、逐条预测文件，以及 `custom_model.joblib`。手写模型脚本会重新加载保存的模型，核对测试概率完全一致。记录级文件和模型二进制文件均由 Git 忽略。原始 CSV 不会被覆盖。

## 后续工作

1. 使用交叉验证比较算法和参数，分析清洗策略与类别表现。
2. 比较单树、Bagging、随机特征和不同树数量，完成消融与效率实验。
3. 核对 Kaggle 标签编码，整理最终预测和小组报告材料。
