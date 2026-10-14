"""固定三组候选参数，运行五折分层交叉验证并逐次保存记录。"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import hashlib
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from common.preprocessing import clean_headers
from common.prepare_features import DATA_DIR, RANDOM_STATE, build_tfidf


MODELS = {
    "naive_bayes": MultinomialNB(alpha=0.1),
    "linear_svc": SVC(kernel="linear", C=10, random_state=RANDOM_STATE),
    "rbf_svc": SVC(kernel="rbf", C=10, gamma=0.1, random_state=RANDOM_STATE),
}


def main():
    # 稳定性检查只读取有标签数据。
    data_path = DATA_DIR / "train_data.csv"
    data = pd.read_csv(data_path)
    assert not data[["text", "target"]].isna().any().any()
    texts = data.text.map(clean_headers)  # 固定的逐篇规则，不学习跨文档统计量。
    labels = data.target
    out = EXPERIMENT_RESULTS / datetime.now().strftime("stability_5fold_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(), "status": "running",
        "cv": {"class": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": RANDOM_STATE},
        "preprocessing": "clean_headers; uncertain headers kept unchanged",
        "models": {name: model.get_params() for name, model in MODELS.items()},
        "train_csv_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "std_definition": "sample standard deviation across 5 folds (ddof=1); not a confidence interval",
        "caveat": "Candidates were selected using an earlier validation split. This is a stability check, not independent or nested model evaluation.",
        "timing": "TF-IDF shared by 3 models within each fold; do not sum feature_seconds three times.",
    }

    def save_metadata():
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def log(message):
        print(message, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(datetime.now().astimezone().isoformat() + " " + message + "\n")

    save_metadata()
    for filename in (Path(__file__).name, "prepare_features.py", "compare_header_cleaning.py"):
        (out / filename).write_bytes(source_path(filename).read_bytes())
        snapshot_common(out)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(texts, labels))
    assignments = pd.DataFrame({"row_index": data.index, "target": labels, "validation_fold": 0})
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        assert not np.intersect1d(train_idx, val_idx).size
        assignments.loc[val_idx, "validation_fold"] = fold
    assert (assignments.validation_fold > 0).all()
    assignments.to_csv(out / "fold_assignments.csv", index=False)
    # 每条样本的验证折已给出；该折以外的所有样本即其训练集。
    oof = assignments.copy()
    for name in MODELS:
        oof[name] = pd.Series(pd.NA, index=oof.index, dtype="Int64")
    rows, class_rows = [], []
    log(f"记录目录：{out}")
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        started = perf_counter()
        vectorizer, X_train, X_val = build_tfidf(texts.iloc[train_idx], texts.iloc[val_idx], ngram_range=(1, 1), min_df=1)
        feature_seconds = perf_counter() - started
        metadata["tfidf_parameters"] = vectorizer.get_params()
        save_metadata()
        log(f"第 {fold}/5 折，训练 {len(train_idx)}，验证 {len(val_idx)}，词表 {len(vectorizer.vocabulary_)}")
        for name, prototype in MODELS.items():
            model = clone(prototype)
            log(f"开始 {name}")
            started = perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model.fit(X_train, labels.iloc[train_idx])
            fit_seconds = perf_counter() - started
            started = perf_counter()
            pred = model.predict(X_val)
            predict_seconds = perf_counter() - started
            row = {
                "fold": fold, "model": name,
                "train_rows": len(train_idx), "validation_rows": len(val_idx),
                "accuracy": accuracy_score(labels.iloc[val_idx], pred),
                "macro_f1": f1_score(labels.iloc[val_idx], pred, average="macro"),
                "correct": int((pred == labels.iloc[val_idx].to_numpy()).sum()),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "feature_seconds": feature_seconds, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "fit_status": int(model.fit_status_) if hasattr(model, "fit_status_") else None,
                "effective_gamma": float(model._gamma) if name == "rbf_svc" else None,
                "warnings": " | ".join(str(w.message) for w in captured),
            }
            rows.append(row)
            oof.loc[val_idx, name] = pred
            report = classification_report(labels.iloc[val_idx], pred, labels=list(range(10)), output_dict=True, zero_division=0)
            class_rows.extend({"fold": fold, "model": name, "label": label, **report[str(label)]} for label in range(10))
            pd.DataFrame(rows).to_csv(out / "fold_results.csv", index=False)
            pd.DataFrame(class_rows).to_csv(out / "per_class_metrics.csv", index=False)
            oof.to_csv(out / "oof_predictions.csv", index=False)
            log(f"完成 {len(rows)}/15：{name} Accuracy={row['accuracy']:.4%}, Macro-F1={row['macro_f1']:.4f}, 训练={fit_seconds:.2f}s")

    results = pd.DataFrame(rows)
    assert len(results) == 15 and not oof[list(MODELS)].isna().any().any()
    summaries = []
    for name in MODELS:
        selected = results[results.model == name]
        for fold in range(1, 6):
            subset = oof[oof.validation_fold == fold]
            score = selected[selected.fold == fold].iloc[0]
            assert abs(accuracy_score(subset.target, subset[name].astype(int)) - score.accuracy) < 1e-12
            assert abs(f1_score(subset.target, subset[name].astype(int), average="macro") - score.macro_f1) < 1e-12
        summaries.append({"model": name,
            "accuracy_mean": selected.accuracy.mean(), "accuracy_std": selected.accuracy.std(ddof=1),
            "accuracy_min": selected.accuracy.min(), "accuracy_max": selected.accuracy.max(),
            "macro_f1_mean": selected.macro_f1.mean(), "macro_f1_std": selected.macro_f1.std(ddof=1),
            "fit_seconds_mean": selected.fit_seconds.mean(), "predict_seconds_mean": selected.predict_seconds.mean(),
            "oof_accuracy": accuracy_score(oof.target, oof[name].astype(int)),
            "oof_macro_f1": f1_score(oof.target, oof[name].astype(int), average="macro"),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary.csv", index=False)
    paired = results.pivot(index="fold", columns="model", values="accuracy")
    paired["linear_minus_nb_pp"] = 100 * (paired.linear_svc - paired.naive_bayes)
    paired["linear_minus_rbf_pp"] = 100 * (paired.linear_svc - paired.rbf_svc)
    paired.to_csv(out / "paired_comparison.csv")
    lines = ["# 五折分层交叉验证：固定候选的稳定性", "",
        "三模型共用五折，每折重新拟合 TF-IDF（单词、min_df=1）；邮件头清洗；随机种子42。",
        "标准差采用样本标准差（ddof=1），描述五折波动，不是置信区间。", "",
        "| 模型 | 平均 Accuracy | 标准差（百分点） | 最低 Accuracy | 最高 Accuracy | 平均 Macro-F1 | F1 标准差 | 平均训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples(index=False):
        lines.append(f"| {r.model} | {r.accuracy_mean:.4%} | {r.accuracy_std*100:.4f} | {r.accuracy_min:.4%} | {r.accuracy_max:.4%} | {r.macro_f1_mean:.6f} | {r.macro_f1_std:.6f} | {r.fit_seconds_mean:.3f} |")
    lines += ["", "## 逐折 Accuracy", "", "| 折 | NB | 线性 SVC | RBF SVC | 线性−NB（百分点） | 线性−RBF（百分点） |", "|---:|---:|---:|---:|---:|---:|"]
    for fold, r in paired.iterrows():
        lines.append(f"| {fold} | {r.naive_bayes:.4%} | {r.linear_svc:.4%} | {r.rbf_svc:.4%} | {r.linear_minus_nb_pp:+.4f} | {r.linear_minus_rbf_pp:+.4f} |")
    lines += ["", "参数已参考过先前验证集，因此此处用于稳定性检查，不是独立或嵌套评估。五折训练集相互重叠，不能把这些分数当作五次独立试验。",
              "summary.csv 同时记录整体折外预测指标（OOF）；整体 OOF Macro-F1 与逐折 Macro-F1 的均值定义不同。",
              "fold_assignments.csv 记录每条数据的验证折，其他四折构成其训练集；oof_predictions.csv 为每条样本仅在其验证折生成的预测。",
              "fit_status 为0表示SVC正常完成；NB无此字段。警告保存在fold_results.csv。"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata["status"] = "complete"
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    save_metadata()
    log("完成：15组结果及折外预测评分校验通过。")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
