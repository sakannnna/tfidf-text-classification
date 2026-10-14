"""分析已保存的验证预测，不重新训练、不使用无标签测试集。"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import hashlib
import json
from datetime import datetime

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from common.prepare_features import DATA_DIR
from common.preprocessing import clean_headers, parse_header


LINEAR_RUN = EXPERIMENT_RESULTS / "svc_grid_20260921_134017_563190"
RBF_RUN = EXPERIMENT_RESULTS / "svc_rbf_grid_20260921_135028_241491"
TOPICS = ["图形学", "Windows", "PC硬件", "汽车", "摩托车", "棒球", "密码学", "电子学", "医学", "中东政治"]


def main():
    data = pd.read_csv(DATA_DIR / "train_data.csv")
    linear = pd.read_csv(LINEAR_RUN / "validation_predictions.csv")
    rbf = pd.read_csv(RBF_RUN / "validation_predictions.csv")
    digest = hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest()
    for path in (LINEAR_RUN, RBF_RUN):
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["train_csv_sha256"] == digest, "源数据与训练记录不一致"
    assert linear.row_index.is_unique and rbf.row_index.is_unique
    joined = linear[["row_index", "target", "C_10.0"]].merge(
        rbf[["row_index", "target", "C_10.0_gamma_0.1"]],
        on="row_index", validate="one_to_one", suffixes=("_linear", "_rbf"),
    )
    assert len(joined) == len(linear) == len(rbf)
    assert joined.target_linear.equals(joined.target_rbf)
    joined = joined.rename(columns={"target_linear": "target", "C_10.0": "linear_prediction", "C_10.0_gamma_0.1": "rbf_prediction"}).drop(columns="target_rbf")
    assert (data.loc[joined.row_index, "target"].to_numpy() == joined.target.to_numpy()).all()
    joined["text"] = data.loc[joined.row_index, "text"].to_numpy()
    joined["cleaned_text"] = joined.text.map(clean_headers)
    joined["subject"] = joined.text.map(lambda t: " | ".join(v for k, v in parse_header(t)[0] if k == "subject") if parse_header(t) else "[未解析]")
    joined["linear_correct"] = joined.target == joined.linear_prediction
    joined["rbf_correct"] = joined.target == joined.rbf_prediction
    out = AUDIT_RESULTS / datetime.now().strftime("error_analysis_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    errors = joined.loc[~joined.linear_correct].copy()
    errors.to_csv(out / "linear_errors.csv", index=False)
    joined.loc[joined.linear_correct != joined.rbf_correct].to_csv(out / "model_differences.csv", index=False)
    labels = list(range(10))
    cm = confusion_matrix(joined.target, joined.linear_prediction, labels=labels)
    pd.DataFrame(cm, index=pd.Index(labels, name="true_label"), columns=labels).to_csv(out / "confusion_matrix.csv")
    report = classification_report(joined.target, joined.linear_prediction, labels=labels, output_dict=True, zero_division=0)
    per_class = pd.DataFrame([{ "label": i, "topic_inferred": TOPICS[i], **report[str(i)]} for i in labels])
    per_class.to_csv(out / "per_class_metrics.csv", index=False)
    pairs = errors.groupby(["target", "linear_prediction"]).size().reset_index(name="count").sort_values(["count", "target", "linear_prediction"], ascending=[False, True, True])
    pairs.to_csv(out / "confusion_pairs.csv", index=False)
    counts = {
        "both_correct": int((joined.linear_correct & joined.rbf_correct).sum()),
        "linear_only_correct": int((joined.linear_correct & ~joined.rbf_correct).sum()),
        "rbf_only_correct": int((~joined.linear_correct & joined.rbf_correct).sum()),
        "both_wrong": int((~joined.linear_correct & ~joined.rbf_correct).sum()),
    }
    (out / "metadata.json").write_text(json.dumps({"linear_run": LINEAR_RUN.name, "rbf_run": RBF_RUN.name, "train_csv_sha256": digest, "comparison": counts}, indent=2), encoding="utf-8")
    lines = ["# 线性 SVC 错误分析", "", "仅使用已保存的验证预测；类别主题为从文本推断的名称。",
             f"验证 {len(joined)} 条，线性 SVC 错 {len(errors)} 条。", "",
             "| 标签 | 推测主题 | Precision | Recall | F1 | 样本数 |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in per_class.to_dict("records"):
        lines.append(f"| {row['label']} | {row['topic_inferred']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += ["", "## 主要混淆方向", "", "| 真实→预测 | 数量 |", "|---|---:|"]
    for row in pairs.head(10).itertuples(index=False):
        lines.append(f"| {row.target} {TOPICS[row.target]} → {row.linear_prediction} {TOPICS[row.linear_prediction]} | {row.count} |")
    lines += ["", "## 与最佳 RBF 对照", "", *[f"- {k}: {v}" for k,v in counts.items()], "",
              "## 阅读样本", "", "每个前六大混淆方向选原始行号最小的两篇，并附一篇同真实类别的正确样本；选择规则固定，不按解释挑选。原始行号从 0 开始。"]
    selected = []
    for row in pairs.head(6).itertuples(index=False):
        subset = errors[(errors.target == row.target) & (errors.linear_prediction == row.linear_prediction)].sort_values("row_index").head(2)
        selected.extend(subset.row_index.tolist())
        correct = joined[(joined.target == row.target) & joined.linear_correct].sort_values("row_index").head(1)
        selected.extend(correct.row_index.tolist())
    for idx in dict.fromkeys(selected):
        row = joined[joined.row_index == idx].iloc[0]
        lines += ["", f"### 行 {idx}：真实 {row.target}，线性预测 {row.linear_prediction}，RBF预测 {row.rbf_prediction}", "", "````text", row.cleaned_text, "````"]
    (out / "review_samples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    plot = ax.imshow(cm, cmap="Blues")
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=9)
    ax.set(xticks=labels, yticks=labels, xlabel="Predicted label", ylabel="True label", title="Linear SVC (C=10): validation confusion matrix")
    fig.colorbar(plot, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(out)
    print(per_class.to_string(index=False))
    print(pairs.head(10).to_string(index=False))
    print(counts)


if __name__ == "__main__":
    main()
