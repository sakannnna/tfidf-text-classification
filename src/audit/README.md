# 辅助审查（可选）

本目录不在最终训练链路中。

- analyze_errors.py：读取真实验证标签和已保存的SVC预测，生成错例与混淆分析。入口为python run.py errors --execute。
- review_uncertain.py：读取无标签测试文本、历史判读JSON和已拟合TF-IDF，检索相近训练文本。入口为python run.py review-uncertain；只读，不生成真实标签。

这两项脚本用于实验后的错例分析，依赖LINEAR_RUN、RBF_RUN或RUN指向的历史结果目录。相关逐条预测、模型和判读JSON未纳入仓库，因此仅克隆仓库不能直接运行这部分。复现分析需要补齐对应文件，并将上述路径设置为实际结果目录。

无标签文本的辅助判读仅用于理解模型分歧，没有用于训练或模型选择。报告中的成绩和最终方案依据见[验证结果](../../docs/evidence/README.md)。
