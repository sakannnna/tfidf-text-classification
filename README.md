# TF-IDF文本分类实验

使用朴素贝叶斯、SVC和MLP完成10分类，比较特征与模型参数，并检查训练稳定性。最终采用单层100单元MLP，线性SVC作为成本较低的对照。

## 内容索引

1. [结果与曲线](docs/evidence/README.md)：模型对照结果与最终配置的选择依据。
2. [公共模块](src/common/README.md)：数据划分、清洗、TF-IDF和固定参数。
3. [最终流程](src/final/README.md)：训练和生成预测的完整入口。
4. [实验过程](src/experiments/README.md)：调参、五折、种子重复和损失记录。
5. [辅助审查](src/audit/README.md)：解释错例，不参与正式训练。

## 目录

```text
run.py             统一命令入口，训练必须显式加 --execute
requirements.txt   已运行实验的固定依赖版本
src/
  common/          数据处理、参数、模型工厂和基线校验
  final/           最终全量训练与预测
  experiments/     验证实验和损失记录
  audit/           错例与无标签文本辅助复核
  tests/           轻量检查
data/              课程数据存放位置，仓库仅包含说明
results/           运行时生成的结果目录
docs/evidence/     精选验证摘要、曲线和数值记录
```

## 准备与运行

实验环境为Python 3.13.7、scikit-learn 1.9.1，依赖版本固定在requirements.txt中。损失记录使用了sklearn内部接口，复现时需要保持版本一致。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 run.py check
```

按[data说明](data/README.md)放入课程CSV后，以下命令会重新训练两个最终模型：

```powershell
.\.venv\Scripts\python.exe -X utf8 run.py predict --execute
```

输出到results/final/test_predictions_<时间>/。最终提交方案使用MLP预测，对应prediction_mlp.csv，提交文件名为prediction.csv；SVC预测用于对照。两份输出都保持测试顺序，单列整数、无表头和索引。

## 实验结果

在固定验证划分上，NB准确率为93.35%（含双词）、线性SVC为94.03%、最终MLP为94.44%。详细参数和结果见证据目录。MLP三种子重复用于检查固定划分下的训练稳定性，NB/SVC五折用于检查划分变化的影响。上述成绩均来自有标签的验证数据。

仓库中的模型参数与保存的实验记录一致。当前目录结构已通过15项检查，覆盖数据处理、基线校验、运行入口和损失记录钩子，其中钩子测试使用微型合成MLP。完整课程实验的结果保留在证据目录；目录调整后未重新执行全流程训练。

## 仓库内容范围

本仓库包含源码、运行说明、依赖配置、检查脚本和精选验证记录。课程原始数据、模型文件及完整运行产物未纳入仓库，.gitignore中配置了对应规则。复现所需的数据格式见data/README.md，各实验的输入依赖见对应目录说明。

实验报告PDF和最终预测CSV属于课程提交材料，与代码仓库分开提供。
