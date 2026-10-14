"""原始文本与邮件头清洗对照实验。"""
from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
from common.preprocessing import clean_headers, check_cleaning, parse_header, HEADER_SEPARATOR
from common.prepare_features import load_data, split_data, build_tfidf
from experiments.naive_bayes_baseline import train_and_evaluate

def main():
    labeled_df, test_df = load_data()
    check_cleaning(labeled_df["text"], "有标签数据")
    check_cleaning(test_df["text"], "无标签测试数据（仅检查清洗规则）")

    # 展示真实样本的邮件头清洗；正文不打印，避免长文本淹没结果。
    example = labeled_df["text"].iloc[0]
    fields, _ = parse_header(example)
    separator = HEADER_SEPARATOR.search(example)
    print("\n原邮件头：\n" + example[:separator.start()])
    print("\n清洗后保留的标题：")
    print("\n".join(value for key, value in fields if key == "subject"))
    print("（随后接完整原正文）")

    # 只划分一次，保证两组使用完全相同的训练/验证样本与顺序。
    train_texts, validation_texts, y_train, y_validation = split_data(labeled_df)
    results = {}
    for name, transform in (("raw", lambda text: text), ("clean_headers", clean_headers)):
        # 每组独立 fit；不能共享原始文本拟合出的词表和 IDF。
        vectorizer, X_train, X_validation = build_tfidf(
            train_texts.map(transform), validation_texts.map(transform)
        )
        _, predictions, scores = train_and_evaluate(
            X_train, y_train, X_validation, y_validation, alpha=1.0
        )
        results[name] = (predictions, scores)
        print(
            f"\n{name}: 词表={len(vectorizer.vocabulary_)}; "
            f"Accuracy={scores['accuracy']:.4%}; Macro-F1={scores['macro_f1']:.4f}"
        )

    raw_predictions, raw_scores = results["raw"]
    clean_predictions, clean_scores = results["clean_headers"]
    raw_correct = raw_predictions == y_validation.to_numpy()
    clean_correct = clean_predictions == y_validation.to_numpy()
    print(f"\n原来错 → 清洗后对：{int((~raw_correct & clean_correct).sum())}")
    print(f"原来对 → 清洗后错：{int((raw_correct & ~clean_correct).sum())}")
    print(f"Accuracy 变化：{100 * (clean_scores['accuracy'] - raw_scores['accuracy']):+.4f} 个百分点")


if __name__ == "__main__":
    main()
