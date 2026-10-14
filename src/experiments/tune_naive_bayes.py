from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
from experiments.naive_bayes_baseline import train_and_evaluate
"""在固定验证集上比较 12 组参数，并逐组记录结果。

运行：python tune_naive_bayes.py
每次运行创建独立的 results/nb_grid_时间戳 目录，保留历史结果。
"""

import hashlib
import json
import platform
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

import pandas as pd
import sklearn

from common.preprocessing import clean_headers
from common.prepare_features import (
    DATA_DIR, RANDOM_STATE, VALIDATION_SIZE,
    load_data, split_data, build_tfidf,
)


NGRAM_RANGES = [(1, 1), (1, 2)]
MIN_DFS = [1, 2]
ALPHAS = [0.1, 0.5, 1.0]


def main():
    labeled_df, _ = load_data()
    train_texts, validation_texts, y_train, y_validation = split_data(labeled_df)
    train_texts = train_texts.map(clean_headers)
    validation_texts = validation_texts.map(clean_headers)

    run_dir = EXPERIMENT_RESULTS / datetime.now().strftime("nb_grid_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "preprocessing": "clean_headers (uncertain headers retained unchanged)",
        "classifier": "MultinomialNB",
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "train_rows": len(y_train), "validation_rows": len(y_validation),
        "grid": {"ngram_range": NGRAM_RANGES, "min_df": MIN_DFS, "alpha": ALPHAS},
        "ranking": "accuracy descending, then macro_f1 descending",
        "python": platform.python_version(),
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "train_csv_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
        "caveat": "Scores are from a reused validation split, not an independent test set.",
        "timing": "feature_seconds is shared by all three alphas; model_seconds includes fit, predict and scoring.",
    }
    # 保存实际划分和代码快照，便于日后复现本次实验。
    pd.concat([
        pd.DataFrame({"row_index": train_texts.index, "split": "train"}),
        pd.DataFrame({"row_index": validation_texts.index, "split": "validation"}),
    ], ignore_index=True).to_csv(run_dir / "split.csv", index=False)
    for filename in (Path(__file__).name, "prepare_features.py", "compare_header_cleaning.py"):
        (run_dir / filename).write_bytes(source_path(filename).read_bytes())
        snapshot_common(run_dir)

    rows = []
    validation_predictions = pd.DataFrame({
        "row_index": validation_texts.index,
        "target": y_validation.to_numpy(),
    })
    for ngram_range, min_df in product(NGRAM_RANGES, MIN_DFS):
        started = perf_counter()
        vectorizer, X_train, X_validation = build_tfidf(
            train_texts, validation_texts, ngram_range=ngram_range, min_df=min_df
        )
        feature_seconds = perf_counter() - started
        metadata["tfidf_fixed_parameters"] = {
            key: value for key, value in vectorizer.get_params().items()
            if key not in ("ngram_range", "min_df")
        }

        # alpha 不影响 TF-IDF，同一组特征可供三个模型复用。
        for alpha in ALPHAS:
            run_id = f"ngram_{ngram_range[1]}_min_df_{min_df}_alpha_{alpha}"
            started = perf_counter()
            _, predictions, scores = train_and_evaluate(
                X_train, y_train, X_validation, y_validation, alpha=alpha
            )
            model_seconds = perf_counter() - started
            rows.append({
                "run_id": run_id, "ngram_range": str(ngram_range),
                "min_df": min_df, "alpha": alpha,
                **scores, "vocabulary_size": len(vectorizer.vocabulary_),
                "correct": int((predictions == y_validation.to_numpy()).sum()),
                "validation_rows": len(y_validation),
                "feature_seconds": feature_seconds, "model_seconds": model_seconds,
            })
            validation_predictions[run_id] = predictions
            # 每完成一组就记录，运行中断时已完成的结果仍保留。
            pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
            validation_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
            print(f"{len(rows):2}/12 {run_id}: Accuracy={scores['accuracy']:.4%}, Macro-F1={scores['macro_f1']:.4f}", flush=True)

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    ranked = pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1"], ascending=False, kind="stable"
    )
    ranked.to_csv(run_dir / "ranked_results.csv", index=False)
    report = [
        "# TF-IDF + 朴素贝叶斯参数对照", "",
        f"邮件头清洗；随机种子 {RANDOM_STATE}；训练 {len(y_train)} 条，验证 {len(y_validation)} 条。",
        "按 Accuracy 降序排列，同分时按 Macro-F1 排序。耗时单位为秒。", "",
        "| ngram_range | min_df | alpha | Accuracy | Macro-F1 | 词表大小 | 特征耗时 | 模型及评分耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked.itertuples(index=False):
        report.append(f"| {row.ngram_range} | {row.min_df} | {row.alpha} | {row.accuracy:.4%} | {row.macro_f1:.6f} | {row.vocabulary_size} | {row.feature_seconds:.3f} | {row.model_seconds:.3f} |")
    report.extend(["", "同一特征配置的三个 alpha 共用一次特征提取，特征耗时不要重复累加。",
                   "这是复用验证集上的调参结果，不能视为独立测试成绩。未生成测试集预测。"])
    (run_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n记录目录：{run_dir}")
    print("最佳配置：" + ranked.iloc[0]["run_id"])


if __name__ == "__main__":
    main()
