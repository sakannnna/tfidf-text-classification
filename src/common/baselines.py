"""显式基线路径与训练前核验；不训练分类器，不自动选最新目录。"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score

from common.config import DATA_DIR, FINAL_TFIDF, RANDOM_STATE, VALIDATION_SIZE, THREAD_LIMIT
from common.models import make_models
from common.prepare_features import split_data, build_tfidf

def normalized(value):
    return json.loads(json.dumps(value, default=str))

def svc_prediction_column(frame):
    for name in ("C_10.0_gamma_scale", "C_10.0"):
        if name in frame:
            return name
    raise ValueError("SVC基线缺少C=10、gamma=scale预测列（支持新旧命名）")

def validate_baseline(directory, kind, *, data_dir=DATA_DIR):
    """在读取全量特征/建立输出目录/训练之前，验证数据、划分和基线配置。"""
    path = Path(directory).expanduser().resolve()
    def require(ok, message):
        if not ok:
            raise ValueError(f"{kind}基线 {path}: {message}")
    require(kind in ("svc", "mlp"), "未知基线类型")
    required = ["metadata.json", "split.csv", "validation_predictions.csv", "results.csv"]
    if kind == "mlp":
        required.append("loss_curve.csv")
    missing = [name for name in required if not (path / name).is_file()]
    require(not missing, f"缺少文件 {missing}；请先运行{'tune-svc' if kind == 'svc' else 'mlp-baseline'}并指定其完整输出目录")
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    train_path = Path(data_dir) / "train_data.csv"
    require(train_path.is_file(), f"找不到课程数据 {train_path}")
    require(meta.get("train_csv_sha256") == hashlib.sha256(train_path.read_bytes()).hexdigest(), "训练数据指纹不一致")
    require(meta.get("preprocessing") == "clean_headers", "清洗方式不一致")
    require(meta.get("random_state") == RANDOM_STATE and meta.get("validation_size") == VALIDATION_SIZE, "划分种子或验证比例不一致")
    # 小型人工文本仅用于取得与公共函数完全相同的向量器参数，无正式数据拟合。
    vectorizer, _, _ = build_tfidf(["alpha beta"], ["alpha"], **FINAL_TFIDF)
    require(meta.get("tfidf_parameters") == normalized(vectorizer.get_params()), "TF-IDF配置不是锁定的单词min_df=1")
    train = pd.read_csv(train_path)
    tr, va, _, y = split_data(train)
    expected = pd.concat([pd.DataFrame({"row_index":tr.index, "split":"train"}),
                          pd.DataFrame({"row_index":va.index, "split":"validation"})], ignore_index=True)
    require(pd.read_csv(path / "split.csv").equals(expected), "数据划分或行顺序不一致")
    pred = pd.read_csv(path / "validation_predictions.csv")
    require({"row_index", "target"}.issubset(pred.columns), "预测表缺少索引/标签")
    require(pred[["row_index","target"]].equals(pd.DataFrame({"row_index":va.index, "target":y.to_numpy()})), "验证索引或真实标签不一致")
    result = pd.read_csv(path / "results.csv")
    require(not result.empty, "结果为空")
    require(meta.get("status", "complete") == "complete", "运行未完成")
    if kind == "mlp":
        require(meta.get("parameters") == normalized(make_models()["mlp"].get_params()), "MLP基线必须为锁定的(100,)、alpha=0.0001、种子42配置")
        for key, value in {"python":platform.python_version(), "sklearn":sklearn.__version__, "numpy":np.__version__, "pandas":pd.__version__, "thread_limit":THREAD_LIMIT}.items():
            require(meta.get(key) == value, f"环境或线程配置不一致: {key}")
        column = "prediction"
        curve = pd.read_csv(path / "loss_curve.csv")
        require(len(result) == 1 and "epochs" in result, "MLP结果表结构不完整")
        require({"accuracy","macro_f1","train_accuracy","fit_seconds","parameter_count","warnings"}.issubset(result.columns), "MLP结果表缺少必要指标")
        require(result.warnings.fillna("").eq("").all(), "MLP基线含警告，请检查后另选完整基线")
        require(len(curve) == int(result.iloc[0]["epochs"]) and "training_loss" in curve and np.isfinite(curve.training_loss).all(), "MLP损失记录不完整")
        selected = result.iloc[0]
    else:
        fixed = meta.get("svc_fixed_parameters", {})
        expected_fixed = normalized(make_models()["svc"].get_params())
        require(all(fixed.get(k) == v for k,v in expected_fixed.items() if k not in ("C","gamma")), "SVC固定配置不一致")
        require(meta.get("kernel", fixed.get("kernel")) == "linear", "SVC必须使用linear核")
        require(10.0 in meta.get("C_values", []) and meta.get("gamma_values", ["scale"]) == ["scale"], "需要包含C=10、gamma=scale的线性基线")
        require("fit_status" in result and result.fit_status.eq(0).all(), "SVC未正常收敛")
        require(len(result) == len(meta["C_values"]), "SVC网格结果不完整")
        require("C" in result and sorted(result.C.tolist()) == sorted(meta["C_values"]), "SVC结果C配置不完整")
        require({"accuracy","macro_f1"}.issubset(result.columns), "SVC结果缺少评分")
        selected = result.loc[result.C.eq(10.0)].iloc[0]
        column = svc_prediction_column(pred)
    require(column in pred, "预测列不存在")
    require(pred[column].isin(train.target.unique()).all(), "预测标签无效")
    require(abs(accuracy_score(pred.target, pred[column])-selected["accuracy"]) < 1e-12, "保存的准确率与预测不一致")
    require(abs(f1_score(pred.target, pred[column], average="macro")-selected["macro_f1"]) < 1e-12, "保存的F1与预测不一致")
    return path

def baseline_arguments(*, svc=False, mlp=False, resume=False):
    parser = argparse.ArgumentParser(description="显式指定基线；先核验，再训练。路径相对当前工作目录。")
    if svc:
        parser.add_argument("--svc-baseline", required=True, type=Path)
    if mlp:
        parser.add_argument("--mlp-baseline", required=True, type=Path)
    if resume:
        parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-only", action="store_true", help="仅核验依赖，不创建结果或训练")
    args = parser.parse_args()
    try:
        if svc:
            args.svc_baseline = validate_baseline(args.svc_baseline, "svc")
        if mlp:
            args.mlp_baseline = validate_baseline(args.mlp_baseline, "mlp")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.check_only:
        print("基线路径、数据指纹、划分、配置和结果完整性核验通过；未训练。")
    return args
