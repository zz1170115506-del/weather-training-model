# Task1 评分与交付检查表

依据：用户提供的课程项目 PDF，2026-11-03 核对。仅追踪 Dataset1 及相关全组交付接口。勾选表示完成并有证据，计划本身不构成证据。

## 评分映射

| 项目 | 课程明示 | A 档应准备的具体证据 | 当前状态 |
| --- | --- | --- | --- |
| Code | 总计 6%；质量、技术设计、测试表现 | 模块清楚、注释/命名有帮助，自主算法，可运行入口，指标/时间 | 自主 CART/RF 及测试已有，完整流程待补 |
| Code quality | 评分表单列 3% | 代码组织良好，可读、易维护、可复现 | 需补 CV/提交流程的工程验收 |
| Report | 5%；全面、深入、具体事实支持 | 数据发现、公式/实现、统一对照/消融、错误/效率、结论/局限 | 尚未形成完整 Task1 章节 |
| Presentation | 4%；非常清晰且有逻辑 | 精简问题/设计/证据/结论，能回答设计与泛化问题 | 待与全组协调 |

表中没有明确的 Task1 独立分值、名次计分公式或“性能 3%”规则。不要自行把推测写成老师规定。

## 任务、数据与自主设计

- [x] 已确认 Task1 为天气四分类和 Macro-F1，映射为 1 Sunny / 2 Cloudy / 3 Rainy / 4 Snowy。
- [x] 已有数据审计及源文件 SHA256；本地三个 CSV 已与已跟踪审计完整核对。
- [x] 已有 NumPy 自主 CART、Bootstrap/随机特征森林、OOB；核心训练预测不调用库分类器。
- [x] 已有单次 Holdout、参数匹配库对照、算法正确性测试及模型重载检查代码。
- [x] 2026-11-03 本地执行 `python -m unittest discover -s code/tests -v`，现有 12 项测试通过；后续改动仍须加入并执行必要流程检查。
- [ ] 核实类别格式、队名、时区及当前 Kaggle 规则，保存确认依据。
- [ ] 补齐统一 5 折、每条恰好一次 OOF、折内预处理和固定概率列顺序。
- [ ] 超参数搜索、少量额外 seeds、最终外层评估及偏差说明。
- [ ] 主要清洗/特征/森林机制/融合消融；保留失败实验结论。
- [ ] 最弱类别、分组样本数、混淆矩阵、互补错误分析。
- [ ] 自主部分在最终方案中的作用清楚，去掉该组件的效果有记录。

## 效率、复现与 Kaggle

- [ ] CPU 型号、内存、服务器数、CPU/数值库线程数、GPU（未使用也说明）。
- [ ] 预处理、模型 fit、总离线时间、全部测试推理时间，注明硬件和计时边界。
- [ ] 树的 epoch 写不适用；迭代模型报告实际迭代数/口径，不编造时间。
- [ ] 区分搜索耗时、CV 耗时、最终训练耗时，避免重复计费或不公平比较。
- [ ] 正式提交校验：`id,label`、完整唯一 ID、样例排序、3,300 行、合法标签、无缺失。
- [ ] 验证平台接受且产生真实评分；保存每份提交的再生成配方与哈希。
- [ ] Public 只作为约一半测试数据的临时结果，选模主要依据稳健本地验证。
- [ ] 每天不超过 10 次，结束前主动选中两份；记录选中的提交编号与理由。
- [ ] 所有实际 Kaggle 提交（尤其所选两份）均可从交付代码、依赖和配置生成；按课程要求，无法生成的对应部分为 0 分。
- [ ] 另一位协作者在干净环境复跑并核对最终预测/CSV 哈希。
- [ ] 最终把真实 Private Macro-F1/名次与本地估计、公榜分开报告。

## 报告、演示和全组打包

- [ ] 全组队名、成员姓名和学号正确。
- [ ] Task1 的问题/数据分析/自主细节/对照消融/效率/局限完整。
- [ ] 按实际工作填写全组贡献表，总计 100%；没有贡献表不会被评分。
- [ ] 参考文献与代码/借鉴算法对应，区分自主实现与库组件。
- [ ] 全组幻灯片 5-10 页，约 7 分钟讲述及 3 分钟问答，Task1 时长与 Task2 协调。
- [ ] 演示能解释：为什么 Macro-F1、为什么这样划分、为什么这样清洗、自主贡献、如何防止公榜过拟合。
- [ ] 本地准备 `TeamName/code/`、`TeamName/report.*`、`TeamName/slides.*`，压缩为 `TeamName.zip`。
- [ ] 由 TEAM COORDINATOR 在官方确认的 2026-11-16 08:00 截止前提交。

预测截止为 PDF 所列 2026-11-14 23:55，排名发布为 11-15 08:00；PDF 未标时区，以 Kaggle/Canvas 为准。以上演示/报告/打包属于全组要求，Task1 文档不能替代 Task2 工作。

## 实验/提交台账字段

在产生真实实验后再建立小型汇总文件，不填造假的分数或 Kaggle 提交编号：

```text
run_id, timestamp_with_timezone, author, code_commit,
config_path, config_sha256, train_sha256, test_sha256, sample_sha256,
fold_rule, fold_hash, seed, model_family, search_budget,
cv_macro_f1_mean, cv_macro_f1_std, oof_macro_f1, evaluation_scope,
per_class_metrics_path, preprocessing_seconds, fit_seconds,
offline_seconds, full_test_inference_seconds, hardware_summary_path,
label_format, prediction_csv_sha256, reproduce_command,
kaggle_submission_id, public_macro_f1, selected_for_final, selection_reason,
private_macro_f1_after_close
```

汇总不含逐行 ID/特征/预测、凭证、个人绝对路径。每份最终 CSV 的权重、种子、Early stopping 迭代数等必须在配置或元数据中充分记录，不能只保存一个分数。
