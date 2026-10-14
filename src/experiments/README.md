# 实验过程

均从仓库根使用run.py。除inspect-data和--check-only外，以下任务需要--execute；每次写入独立results/experiments/目录。

| 任务 | 检查的问题 |
|---|---|
| inspect-data | 长度、类别分布等数据概览，不训练 |
| nb-baseline / compare-cleaning | 跑通基线、邮件头清洗是否有效 |
| tune-nb | 单词/双词、min_df与NB alpha |
| tune-svc / tune-rbf | C、kernel、gamma |
| compare-ngrams | SVC加入双词是否改善 |
| stability | NB及两种SVC在相同五折上的表现 |
| mlp-baseline / tune-mlp / mlp-depth | 单层基线、宽度、正则化及双层网络 |
| mlp-seeds | 固定划分上的三种子重复 |
| mlp-epochs | 原始小批次损失、逐轮训练与验证指标 |

## 基线依赖与复现顺序

1. `python run.py tune-svc --execute`，生成SVC基线结果目录。
2. `python run.py mlp-baseline --execute --svc-baseline "<SVC目录>"`，使用上一步的目录生成MLP基线。
3. 后续任务显式传基线路径，不会自动挑选最新目录。

| 后续任务 | 必须提供 |
|---|---|
| compare-ngrams | --svc-baseline |
| tune-mlp / mlp-depth / mlp-seeds | --svc-baseline及--mlp-baseline |
| mlp-epochs | --mlp-baseline |

这些任务可用--check-only替换--execute先校验，不训练、不创建结果目录。其他表中任务不需要上述基线参数。tune-mlp可加--resume "<目录>"，仅接受源码快照及公共模块一致的运行，不混用旧版本快照。

## 损失口径

mlp-epochs直接记录库每批次更新前返回的训练损失（含L2），并在每轮末计算不含L2的训练/验证交叉熵。记录不改变梯度、优化器和停止条件。验证集仅评分。保存batch_metrics.csv、epoch_metrics.csv及曲线；没有移动平均、拟合或下采样。

plot_mlp_batches.py可对已有结果仅绘图：`python src/experiments/plot_mlp_batches.py "<结果目录>"`。钩子依赖sklearn私有接口，升级后必须重新检查。

各阶段的结果摘要位于[证据目录](../../docs/evidence/README.md)，保留原实验数值和来源信息。
