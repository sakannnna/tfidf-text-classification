"""文档算法二：固定 TF-IDF，比较线性核 SVC 的 C=0.1、1、10。"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import hashlib
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from common.preprocessing import clean_headers
from common.prepare_features import DATA_DIR, RANDOM_STATE, VALIDATION_SIZE, load_data, split_data, build_tfidf


C_VALUES = [0.1, 1.0, 10.0]


def run_experiment(kernel="linear", c_values=None, gamma_values=None, ngram_range=(1, 1)):
    """固定数据和特征，逐组运行 SVC；默认保留原来的线性核实验。"""
    c_values = C_VALUES if c_values is None else c_values
    gamma_values = ["scale"] if gamma_values is None else gamma_values
    labeled_df, _ = load_data()
    train_texts, validation_texts, y_train, y_validation = split_data(labeled_df)
    started = perf_counter()
    vectorizer, X_train, X_validation = build_tfidf(
        train_texts.map(clean_headers), validation_texts.map(clean_headers),
        ngram_range=ngram_range, min_df=1,
    )
    feature_seconds = perf_counter() - started
    run_dir = EXPERIMENT_RESULTS / datetime.now().strftime(f"svc_{kernel}_grid_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "classifier": "sklearn.svm.SVC", "preprocessing": "clean_headers",
        "kernel": kernel, "C_values": c_values, "gamma_values": gamma_values, "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "train_rows": len(y_train), "validation_rows": len(y_validation),
        "tfidf_parameters": vectorizer.get_params(),
        "svc_fixed_parameters": {k: v for k, v in SVC(kernel=kernel, random_state=RANDOM_STATE).get_params().items() if k not in ("C", "gamma")},
        "feature_seconds": feature_seconds,
        "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "train_csv_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
        "ranking": "accuracy descending, then macro_f1 descending",
        "caveat": "Reused validation split, not independent test performance. Feature extraction is shared across C values.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.concat([
        pd.DataFrame({"row_index": train_texts.index, "split": "train"}),
        pd.DataFrame({"row_index": validation_texts.index, "split": "validation"}),
    ], ignore_index=True).to_csv(run_dir / "split.csv", index=False)
    for filename in (Path(__file__).name, "tune_rbf_svc.py", "prepare_features.py", "compare_header_cleaning.py"):
        (run_dir / filename).write_bytes(source_path(filename).read_bytes())
        snapshot_common(run_dir)

    rows = []
    predictions_table = pd.DataFrame({"row_index": validation_texts.index, "target": y_validation.to_numpy()})
    print(f"记录目录：{run_dir}\n词表：{len(vectorizer.vocabulary_)}", flush=True)
    configurations = [(C, gamma) for C in c_values for gamma in gamma_values]
    for C, gamma in configurations:
        run_id = f"C_{C}_gamma_{gamma}"
        print(f"开始训练 SVC(kernel={kernel!r}, C={C}, gamma={gamma!r})", flush=True)
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=RANDOM_STATE)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        predictions = model.predict(X_validation)
        predict_seconds = perf_counter() - started
        row = {
            "run_id": run_id, "C": C, "gamma": gamma,
            # sklearn 在 fit 时解析 scale；线性核不使用 gamma。
            "effective_gamma": float(model._gamma) if kernel == "rbf" else None,
            "kernel": kernel, "ngram_range": str(ngram_range), "min_df": 1,
            "accuracy": accuracy_score(y_validation, predictions),
            "macro_f1": f1_score(y_validation, predictions, average="macro"),
            "correct": int((predictions == y_validation.to_numpy()).sum()),
            "validation_rows": len(y_validation), "vocabulary_size": len(vectorizer.vocabulary_),
            "support_vectors": int(model.n_support_.sum()),
            "fit_status": int(model.fit_status_),
            "feature_seconds": feature_seconds, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "warnings": " | ".join(str(w.message) for w in captured),
        }
        rows.append(row)
        predictions_table[run_id] = predictions
        pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
        predictions_table.to_csv(run_dir / "validation_predictions.csv", index=False)
        print(f"完成 {len(rows)}/{len(configurations)} {run_id}: Accuracy={row['accuracy']:.4%}, Macro-F1={row['macro_f1']:.4f}, 训练={fit_seconds:.2f}s, fit_status={model.fit_status_}", flush=True)

    ranked = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False, kind="stable")
    ranked.to_csv(run_dir / "ranked_results.csv", index=False)
    report = [f"# {kernel} 核 SVC 参数对照", "",
              f"清洗邮件头；TF-IDF ngram_range={ngram_range}，min_df=1；分层划分，random_state=42。",
              "训练 5894 条，验证 1474 条。按 Accuracy、Macro-F1 降序排列。", "",
              "| C | gamma | 实际 gamma（仅 RBF） | Accuracy | Macro-F1 | 支持向量数 | 训练秒数 | 预测秒数 | fit_status |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ranked.itertuples(index=False):
        report.append(f"| {row.C} | {row.gamma} | {row.effective_gamma} | {row.accuracy:.4%} | {row.macro_f1:.6f} | {row.support_vectors} | {row.fit_seconds:.3f} | {row.predict_seconds:.3f} | {row.fit_status} |")
    report.extend(["", "fit_status=0 表示正常完成求解；警告详见 results.csv。",
                   "这是复用验证集上的调参结果，不能当作独立测试成绩。"])
    (run_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    run_experiment()
