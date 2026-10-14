# 最终训练与预测

入口：在仓库根运行 `python run.py predict --execute`。

predict_test.py依次读取数据、清洗邮件、在全部有标签文本上拟合TF-IDF、转换测试文本、训练两种模型并生成预测。无需历史实验目录。

- SVC：线性核，C=10。
- MLP：隐藏层(100,)，alpha=0.0001，ReLU/Adam，种子42，最多300轮，按训练损失停止。

配置由common/config.py统一管理。结果保存至results/final/的独立时间戳目录，包括两份预测、模型和元数据。最终提交方案使用prediction_mlp.csv，提交文件名为prediction.csv；两种模型的预测分别保存，没有混合。生成的模型和大型运行产物未纳入仓库。
