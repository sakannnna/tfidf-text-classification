"""公共路径与固定参数；实验网格仍由各实验脚本显式定义。"""
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SOURCE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
THREAD_LIMIT = 4
FINAL_TFIDF = {"ngram_range": (1, 1), "min_df": 1}
FINAL_SVC = {"kernel": "linear", "C": 10, "random_state": RANDOM_STATE}
FINAL_MLP = {
    "hidden_layer_sizes": (100,), "alpha": 0.0001,
    "activation": "relu", "solver": "adam", "max_iter": 300,
    "random_state": RANDOM_STATE, "early_stopping": False, "verbose": True,
}
FINAL_RESULTS = PROJECT_DIR / "results" / "final"
EXPERIMENT_RESULTS = PROJECT_DIR / "results" / "experiments"
AUDIT_RESULTS = PROJECT_DIR / "results" / "audit"

def source_path(name):
    """供旧快照逻辑定位迁移后的源文件。"""
    paths = list(SOURCE_DIR.glob(f"*/{name}"))
    paths = [p for p in paths if p.parent.name in {"common", "final", "experiments", "audit", "tests"}]
    if len(paths) != 1:
        raise FileNotFoundError(f"源文件不唯一或不存在: {name}: {paths}")
    return paths[0]

def snapshot_common(output):
    """快照中同时保存公共依赖，不能只保留调用入口。"""
    import shutil
    shutil.copytree(SOURCE_DIR / "common", Path(output) / "common",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
