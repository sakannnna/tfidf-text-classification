"""文档算法二的 RBF 核对照：2 个 C × 4 个 gamma，共 8 组。

复用 tune_svc 的划分、特征提取、训练和记录流程。
运行：python tune_rbf_svc.py
"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
from experiments.tune_svc import run_experiment


if __name__ == "__main__":
    run_experiment(
        kernel="rbf",
        c_values=[1.0, 10.0],
        gamma_values=["scale", 0.1, 1.0, 10.0],
    )
