# 公共模块

- config.py：项目、源码、数据与结果路径；固定种子42及最终模型参数。
- prepare_features.py：读取、检查、分层划分和TF-IDF。词表只在训练部分拟合。
- preprocessing.py：识别邮件头，保留标题和正文；无法确认时保留原文。
- models.py：构造最终MLP/SVC，不在导入时训练。
- baselines.py：检查实验依赖的基线目录、数据指纹、划分、参数、版本及保存成绩。
- logging_utils.py：终端和日志双写。

最终TF-IDF使用单词特征、min_df=1。早期实验保留各自的默认值，最终流程显式指定选定参数，以便区分不同阶段的对照设置。各脚本通过本模块定位data/和results/，源码快照从src/读取。
