"""公共数据读取、分层划分与TF-IDF；导入时不训练。
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


from common.config import DATA_DIR, RANDOM_STATE, VALIDATION_SIZE


# 第一部分：读取数据并划分训练集、验证集。
def load_data(data_dir: Path = DATA_DIR):
    """返回有标签的数据表、无标签测试数据表，保留原始行顺序。"""
    data_dir = Path(data_dir)
    labeled_df = pd.read_csv(data_dir / "train_data.csv")
    test_df = pd.read_csv(data_dir / "test_data_unlabeled.csv")

    # 不把缺失值直接转成字符串 "nan"，避免生成无意义的文本特征。
    for name, frame, required in (
        ("训练数据", labeled_df, {"text", "target"}),
        ("测试数据", test_df, {"text"}),
    ):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name}缺少列：{required - set(frame.columns)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name}存在缺失值，请先检查数据")
        if frame["text"].str.strip().eq("").any():
            raise ValueError(f"{name}存在空文本，请先检查数据")

    return labeled_df, test_df


def split_data(labeled_df, validation_size=VALIDATION_SIZE, random_state=RANDOM_STATE):
    """按类别比例划分；返回顺序为训练文本、验证文本、训练标签、验证标签。"""
    return train_test_split(
        labeled_df["text"],
        labeled_df["target"],
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
        stratify=labeled_df["target"],  # 尽量保持两部分中的类别比例一致。
    )


# 第二部分：调用 sklearn 的 TF-IDF，将文本转换为稀疏特征矩阵。
def build_tfidf(train_texts, validation_texts, *, ngram_range=(1, 1), min_df=2):
    """只从训练文本学习词表和 IDF，再用相同规则转换验证文本。"""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=1.0,
        max_features=None,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        stop_words=None,
    )

    # fit_transform：学习训练集的词表、IDF，并生成训练特征。
    X_train = vectorizer.fit_transform(train_texts)
    # transform：使用已有词表、IDF；验证集独有的词会被忽略。
    X_validation = vectorizer.transform(validation_texts)

    # 保持稀疏格式，不调用 toarray()，避免大量零元素占用内存。
    return vectorizer, X_train, X_validation
