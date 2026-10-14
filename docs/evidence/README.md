# 验证结果与训练曲线

| 文件 | 用途 |
|---|---|
| nb.md / svc.md | 参数对照 |
| five-fold.md | NB、线性SVC、RBF SVC的同划分五折 |
| mlp-depth.md | 单层与双层比较 |
| mlp-seeds.md | 固定验证划分上的种子重复 |
| mlp-epochs.md | 逐轮损失与过拟合分析 |
| mlp-batches.md | 补记小批次损失的方法和一致性核验 |

[报告三联图](report_mlp_batch_losses.png)包含原始批次损失、轮末交叉熵、后期验证损失；[批次线性/对数轴](mlp_batch_loss_curves.png)展示同一组原始值。mlp_batch_metrics.csv保留2010条批次损失，mlp_batch_epoch_metrics.csv保留67轮指标。没有平滑，批次损失含L2，轮末交叉熵不含L2。

所有摘要、CSV和图片均取自已完成的实验。source_manifest.json及两个*_source.json记录来源与哈希。摘要中的data-Project1路径对应实验时的本地目录，当前代码统一从data/读取数据、向results/写入结果。完整历史目录和报告组合图的绘图脚本未纳入仓库；本目录保留图片与数值记录。实验运行和批次曲线重绘方法见[src/experiments](../../src/experiments/README.md)。

本目录记录的是有标签数据上的验证结果。邮件正文、逐条人工/AI辅助判读记录和训练模型文件未纳入该目录。
