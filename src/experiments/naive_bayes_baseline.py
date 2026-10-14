"""原始朴素贝叶斯基线，默认参数保持不变。"""
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from common.prepare_features import load_data, split_data, build_tfidf

# 第三部分：用训练特征学习分类器，在验证集上评估。
def train_and_evaluate(X_train, y_train, X_validation, y_validation, alpha=1.0):
    """返回训练好的模型、验证集预测和评分，方便后续对照实验复用。"""
    # alpha 是平滑强度，第一条基线固定为 1.0。
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    # 验证标签仅用于评分，不参与模型训练。
    predictions = model.predict(X_validation)
    scores = {
        "accuracy": accuracy_score(y_validation, predictions),
        # 先计算每个类别的 F1，再等权平均。
        "macro_f1": f1_score(y_validation, predictions, average="macro"),
    }
    return model, predictions, scores


def main():
    labeled_df, test_df = load_data()
    train_texts, validation_texts, y_train, y_validation = split_data(labeled_df)
    vectorizer, X_train, X_validation = build_tfidf(train_texts, validation_texts)

    print(f"训练集：{len(y_train)} 条；验证集：{len(y_validation)} 条")
    print(f"无标签测试集：{len(test_df)} 条（留待最终预测）")
    print(f"词表大小：{len(vectorizer.vocabulary_)}")
    print(f"训练特征矩阵：{X_train.shape}")
    print(f"验证特征矩阵：{X_validation.shape}")
    print("矩阵的行是样本，列是词表中的词。")

    model, predictions, scores = train_and_evaluate(
        X_train, y_train, X_validation, y_validation
    )
    print(f"\n分类器：MultinomialNB(alpha={model.alpha})")
    print(f"验证集 Accuracy：{scores['accuracy']:.4%}")
    print(f"验证集 Macro-F1：{scores['macro_f1']:.4f}")
    print(f"预测正确：{int((predictions == y_validation.to_numpy()).sum())}/{len(y_validation)}")

    # 确定最终方案后，需要用全部 labeled_df 重新拟合 TF-IDF 和分类器，
    # 再 transform(test_df['text']) 并预测；不要复用本次训练折的 IDF。


if __name__ == "__main__":
    main()
