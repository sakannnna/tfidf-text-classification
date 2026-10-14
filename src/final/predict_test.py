"""锁定参数后全量重训MLP与线性SVC，分别输出无表头的测试预测。

不依赖历史实验目录；不重新划分验证集，也不根据测试预测调参。
运行方式见项目根目录 README.md；此入口会全量训练，请勿用于仅检查导入。
"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import contextlib
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

from common.preprocessing import clean_headers, check_cleaning
from common.prepare_features import DATA_DIR, load_data, build_tfidf
from common.logging_utils import Tee


from common.models import make_models
from common.config import FINAL_TFIDF


def save_predictions(path, predictions, labels, expected_rows):
    """保存后回读，核对行数、单列、整数标签和逐行顺序。"""
    predictions = np.asarray(predictions)
    assert predictions.shape == (expected_rows,)
    assert np.isin(predictions, labels).all()
    assert np.array_equal(predictions, predictions.astype(np.int64))
    pd.DataFrame(predictions.astype(np.int64)).to_csv(path, index=False, header=False)
    restored = pd.read_csv(path, header=None)
    assert restored.shape == (expected_rows, 1)
    np.testing.assert_array_equal(restored.iloc[:, 0].to_numpy(), predictions)


def main():
    out = FINAL_RESULTS / datetime.now().strftime('test_predictions_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    models = make_models()
    meta = {
        'status': 'running', 'created_at': datetime.now().astimezone().isoformat(),
        'purpose': 'Full labeled-data refit with frozen parameters; separate predictions, no ensemble',
        'model_parameters': {name: model.get_params() for name, model in models.items()},
        'preprocessing': 'clean_headers: subject and unchanged body; uncertain headers kept unchanged',
        'python': platform.python_version(), 'platform': platform.platform(),
        'sklearn': sklearn.__version__, 'numpy': np.__version__,
        'pandas': pd.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__,
        'thread_limit': THREAD_LIMIT,
        'train_csv_sha256': hashlib.sha256((DATA_DIR / 'train_data.csv').read_bytes()).hexdigest(),
        'test_csv_sha256': hashlib.sha256((DATA_DIR / 'test_data_unlabeled.csv').read_bytes()).hexdigest(),
        'test_order': 'CSV source order, no sorting/shuffling/dropping',
        'caveat': 'No test labels: no test accuracy or F1. Agreement is not correctness. No tuning on test predictions.',
        'runs': {},
    }
    def save_meta():
        (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    save_meta()
    for name in (Path(__file__).name, 'prepare_features.py', 'preprocessing.py', 'models.py', 'logging_utils.py'):
        (out / name).write_bytes(source_path(name).read_bytes())
        snapshot_common(out)
    print(f'输出目录：{out}', flush=True)
    try:
        train, test = load_data()
        check_cleaning(train.text, '全量有标签数据')
        check_cleaning(test.text, '无标签测试数据')
        train_text, test_text = train.text.map(clean_headers), test.text.map(clean_headers)
        assert train_text.index.equals(train.index) and test_text.index.equals(test.index)
        labels = np.sort(train.target.unique())
        began = perf_counter()
        vectorizer, X_train, X_test = build_tfidf(train_text, test_text, **FINAL_TFIDF)
        meta.update(train_rows=len(train), test_rows=len(test), classes=labels.tolist(),
                    tfidf_parameters=vectorizer.get_params(), vocabulary_size=X_train.shape[1],
                    feature_seconds=perf_counter()-began)
        save_meta()
        joblib.dump(vectorizer, out / 'tfidf.joblib', compress=3)
        audit = pd.DataFrame({'test_row_index': test.index.to_numpy()})
        rows = []
        print(f'全量训练{len(train)}条；测试{len(test)}条；词表{X_train.shape[1]}。', flush=True)
        with threadpool_limits(limits=THREAD_LIMIT):
            meta['threadpools'] = threadpool_info()
            save_meta()
            for name, model in models.items():
                print(f'开始全量训练：{name}', flush=True)
                run = {'status': 'running'}
                meta['runs'][name] = run
                save_meta()
                began = perf_counter()
                with (out / f'{name}_training.log').open('w', encoding='utf-8') as log:
                    with contextlib.redirect_stdout(Tee(sys.stdout, log)), warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(X_train, train.target)
                run.update(fit_seconds=perf_counter()-began, warnings=[str(w.message) for w in caught])
                if any('interrupted' in str(w.message).lower() for w in caught):
                    raise RuntimeError(f'{name} fit interrupted; refusing to export incomplete model')
                if name == 'svc':
                    run.update(fit_status=int(model.fit_status_), support_vectors=int(model.n_support_.sum()))
                    assert model.fit_status_ == 0
                else:
                    run.update(epochs=int(model.n_iter_), parameter_count=sum(x.size for x in model.coefs_+model.intercepts_))
                    pd.DataFrame({'epoch':np.arange(1,model.n_iter_+1), 'training_objective':model.loss_curve_}).to_csv(out / 'mlp_loss_curve.csv', index=False)
                began = perf_counter()
                predictions = model.predict(X_test)
                run['predict_seconds'] = perf_counter()-began
                path = out / f'prediction_{name}.csv'
                save_predictions(path, predictions, labels, len(test))
                audit[name] = predictions
                audit.to_csv(out / 'predictions_with_index.csv', index=False)
                # 保存拟合好的模型；重载后重新预测，确认序列化未改变结果。
                model_path = out / f'{name}.joblib'
                joblib.dump(model, model_path, compress=3)
                restored = joblib.load(model_path)
                np.testing.assert_array_equal(restored.predict(X_test), predictions)
                del restored
                run.update(status='complete', prediction_file=path.name,
                           prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           model_file=model_path.name, serialization_verified=True)
                rows.append({'model':name, **run})
                pd.DataFrame(rows).to_csv(out / 'training_results.csv', index=False)
                save_meta()
                print(f'{name}完成，已验证{len(predictions)}条预测：{path.name}', flush=True)
        assert audit.test_row_index.tolist() == test.index.tolist()
        counts = pd.DataFrame({name: audit[name].value_counts().reindex(labels, fill_value=0) for name in models})
        counts.index.name = 'predicted_label'
        counts.to_csv(out / 'predicted_class_counts.csv')
        different = audit.svc.ne(audit.mlp)
        audit.loc[different].to_csv(out / 'model_disagreements.csv', index=False)
        meta.update(status='complete', finished_at=datetime.now().astimezone().isoformat(),
                    disagreement_count=int(different.sum()), agreement_count=int((~different).sum()))
        save_meta()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4))
        loss = pd.read_csv(out / 'mlp_loss_curve.csv')
        ax.plot(loss.epoch, loss.training_objective)
        ax.set(xlabel='Epoch', ylabel='Training objective (including L2)', yscale='log', title='Full-data MLP training')
        fig.tight_layout()
        fig.savefig(out / 'mlp_loss_curve.png', dpi=160)
        plt.close(fig)
        lines = ['# 两模型全量重训与测试集预测', '',
                 f'使用全部{len(train)}条有标签数据；TF-IDF只在这些文本上重新拟合，再转换{len(test)}条测试文本。共享同一清洗与单词TF-IDF（min_df=1、sublinear_tf=True、L2归一化），词表{X_train.shape[1]}。',
                 '参数在预测前锁定：线性SVC C=10；MLP (100,)、alpha=0.0001、ReLU/Adam、种子42、max_iter=300及原训练损失停止规则。没有把历史67轮写死。', '',
                 '| 模型 | 输出 | 训练秒数 | 轮数/支持向量数 |',
                 '|---|---|---:|---:|']
        for name, run in meta['runs'].items():
            lines.append(f"| {name} | {run['prediction_file']} | {run['fit_seconds']:.2f} | {run.get('epochs',run.get('support_vectors'))} |")
        lines += ['', '两份预测均为单列整数、无表头、无索引，行顺序对应原始测试CSV，行数和标签范围已检查。模型序列化重载后的预测与CSV一致。',
                  f"两模型预测一致{meta['agreement_count']}条，不一致{meta['disagreement_count']}条；一致率不是准确率，测试标签不可见，不能评估哪份更准确。", '',
                  '原验证成绩属于历史训练/验证划分，不是这些全量模型的测试成绩。未根据测试预测调参、挑种子或融合。',
                  '正式提交时只选一份预测，改名为prediction.csv；本次未替用户覆盖或选择正式提交文件。',
                  'tfidf.joblib为两模型共享特征器，svc.joblib/mlp.joblib为全量模型；使用时先调用clean_headers再transform。joblib文件只能从可信来源加载。',
                  'metadata.json、training_results.csv、训练日志、MLP损失CSV/图与代码快照记录训练过程；predictions_with_index.csv便于核验顺序，不能作为无表头提交文件。']
        (out / 'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines), flush=True)
    except BaseException as error:
        meta.update(status='interrupted_or_failed', error=repr(error))
        save_meta()
        raise


if __name__ == '__main__':
    main()
