# weather-training-model

数据集 1 的天气四分类实验项目。目前已完成数据检查、可复现的基线比较和测试集预测生成。现有模型用于建立性能参照；自主实现的分类算法、交叉验证和消融实验尚待完成。

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
```

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
    requirements.txt          固定版本的主要依赖
    README.txt                运行细节与本地输出说明
outputs/dataset1_baseline/
    audit.json                数据概况、检查结果和源文件哈希
    details.json              模型参数、各类分数及混淆矩阵
    metrics.csv               基线分数和计时
    data_overview.png          类别分布与百分比异常统计
    baseline_comparison.png    基线比较及最佳模型混淆矩阵
```

运行时还会生成本地环境信息、记录划分和逐条预测文件。原始 CSV 不会被覆盖。

## 后续工作

1. 自主实现 CART 的不纯度计算、切分搜索、停止条件和叶节点预测。
2. 扩展 Bootstrap 抽样、随机候选特征与多树集成。
3. 使用交叉验证比较算法和参数，分析清洗策略与类别表现。
4. 完成消融实验、运行时间分析及小组报告材料。
