"""文档算法四：MLP 基线；保留同样的数据划分和单词 TF-IDF。"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import contextlib
import hashlib
from common.baselines import baseline_arguments, validate_baseline, svc_prediction_column
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from common.models import make_mlp
from threadpoolctl import threadpool_limits, threadpool_info

from common.preprocessing import clean_headers
from common.prepare_features import DATA_DIR, RANDOM_STATE, load_data, split_data, build_tfidf


from common.logging_utils import Tee

def run_mlp(hidden_layer_sizes=(100,), alpha=0.0001, output_dir=None, *, model_seed=RANDOM_STATE, svc_baseline):
    """运行一组配置；默认参数与最初的 MLP 基线一致。"""
    prior = validate_baseline(svc_baseline, "svc")
    data, _ = load_data()
    train_texts, val_texts, y_train, y_val = split_data(data)
    started = perf_counter()
    vectorizer, X_train, X_val = build_tfidf(train_texts.map(clean_headers), val_texts.map(clean_headers), ngram_range=(1, 1), min_df=1)
    feature_seconds = perf_counter() - started
    out = Path(output_dir) if output_dir is not None else EXPERIMENT_RESULTS / datetime.now().strftime("mlp_baseline_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    model = make_mlp(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, random_state=model_seed)
    meta = {
        "status": "running", "created_at": datetime.now().astimezone().isoformat(),
        "model": "MLPClassifier", "parameters": model.get_params(),
        "tfidf_parameters": vectorizer.get_params(), "preprocessing": "clean_headers",
        "random_state": model_seed, "model_seed": model_seed,
        "split_seed": RANDOM_STATE, "validation_size": 0.2,
        "train_rows": len(y_train), "validation_rows": len(y_val),
        "vocabulary_size": len(vectorizer.vocabulary_), "feature_seconds": feature_seconds,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
        "train_csv_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
        'thread_limit': THREAD_LIMIT,
        "stopping": "early_stopping=False: stopping uses training loss, not external validation scores",
        "loss_definition": "sklearn loss_curve_: epoch training objective including L2 penalty; not validation loss",
        "caveat": "Fixed validation split used for prior model selection. One model seed; not MLP stability evaluation.",
    }
    def save_meta():
        (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    save_meta()
    split = pd.concat([pd.DataFrame({'row_index': train_texts.index, 'split':'train'}), pd.DataFrame({'row_index':val_texts.index,'split':'validation'})], ignore_index=True)
    split.to_csv(out / 'split.csv',index=False)
    assert split.equals(pd.read_csv(prior / 'split.csv'))
    old_meta = json.loads((prior / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['train_csv_sha256'] == old_meta['train_csv_sha256']
    assert json.loads(json.dumps(vectorizer.get_params(),default=str)) == old_meta['tfidf_parameters']
    for filename in (Path(__file__).name, 'prepare_features.py', 'compare_header_cleaning.py'):
        (out / filename).write_bytes(source_path(filename).read_bytes())
        snapshot_common(out)
    print(f'记录目录：{out}\n特征数：{X_train.shape[1]}；隐藏层：{hidden_layer_sizes}；alpha={alpha}；最多300轮', flush=True)
    started = perf_counter()
    # 限制底层计算线程，避免大量线程争用；保持原始稀疏输入和模型参数。
    with threadpool_limits(limits=THREAD_LIMIT), (out / 'training.log').open('w',encoding='utf-8') as stream:
        meta['threadpools'] = threadpool_info()
        save_meta()
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started
    with threadpool_limits(limits=THREAD_LIMIT):
        started = perf_counter()
        pred = model.predict(X_val)
        predict_seconds = perf_counter() - started
        train_pred = model.predict(X_train)
    results = {
        'accuracy': accuracy_score(y_val,pred), 'macro_f1': f1_score(y_val,pred,average='macro'),
        'train_accuracy': accuracy_score(y_train,train_pred), 'train_macro_f1': f1_score(y_train,train_pred,average='macro'),
        'correct':int((pred==y_val.to_numpy()).sum()), 'epochs':model.n_iter_,
        'fit_seconds':fit_seconds, 'predict_seconds':predict_seconds,
        'final_training_loss':model.loss_, 'parameter_count':sum(x.size for x in model.coefs_+model.intercepts_),
        'warnings':' | '.join(str(w.message) for w in captured),
    }
    pd.DataFrame([results]).to_csv(out / 'results.csv',index=False)
    pd.DataFrame({'row_index':val_texts.index,'target':y_val.to_numpy(),'prediction':pred}).to_csv(out / 'validation_predictions.csv',index=False)
    pd.DataFrame({'epoch':range(1,len(model.loss_curve_)+1),'training_loss':model.loss_curve_}).to_csv(out / 'loss_curve.csv',index=False)
    (out / 'classification_report.json').write_text(json.dumps(classification_report(y_val,pred,output_dict=True),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(range(1,len(model.loss_curve_)+1),model.loss_curve_)
    ax.set(xlabel='Epoch',ylabel='Training objective (including L2)',title=f'MLP {hidden_layer_sizes}, alpha={alpha}: training loss')
    fig.tight_layout()
    fig.savefig(out / 'loss_curve.png',dpi=160)
    ax.set_yscale('log')
    fig.savefig(out / 'loss_curve_log.png',dpi=160)
    plt.close(fig)
    meta['status']='complete'
    meta['finished_at']=datetime.now().astimezone().isoformat()
    save_meta()
    report = ['# MLP 实验', '', f'隐藏层{hidden_layer_sizes}；ReLU；Adam；alpha={alpha}；最多300轮；模型种子={model_seed}，划分种子={RANDOM_STATE}。',
              '沿用单词 TF-IDF、min_df=1、邮件头清洗与固定80/20划分。', '',
              f"验证 Accuracy：{results['accuracy']:.4%}；Macro-F1：{results['macro_f1']:.6f}。",
              f"训练 Accuracy：{results['train_accuracy']:.4%}；Macro-F1：{results['train_macro_f1']:.6f}。",
              f"实际训练轮数：{model.n_iter_}；训练耗时：{fit_seconds:.3f}秒；参数量：{results['parameter_count']}。",
              f"训练警告：{results['warnings'] or '无'}。", '',
              '停止依据为训练损失改善（默认tol和n_iter_no_change），未用外部验证集提前停止。',
              'loss_curve.csv/png 保存每轮训练目标，包含正则化项，不是验证损失；训练日志逐轮即时保存。',
              '当前仅一个初始化种子，不代表已验证MLP对初始化的稳定性。']
    (out / 'summary.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report),flush=True)
    return out


def main():
    args = baseline_arguments(svc=True)
    if not args.check_only:
        run_mlp(svc_baseline=args.svc_baseline)


if __name__ == '__main__':
    main()
