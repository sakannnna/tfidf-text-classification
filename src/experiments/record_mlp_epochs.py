"""记录每轮训练/验证指标，保持单层 MLP 原始 fit 和停止规则。

为避免反复fit重置优化器或打乱顺序，使用已核对的sklearn轮末内部钩子。
钩子只做评分和记录；验证数据不参与梯度更新或停止判断。
"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import contextlib
import hashlib
import inspect
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from common.config import FINAL_MLP
from threadpoolctl import threadpool_limits, threadpool_info

from common.preprocessing import clean_headers
from common.prepare_features import DATA_DIR, RANDOM_STATE, load_data, split_data, build_tfidf
from common.logging_utils import Tee




class ObservedMLP(MLPClassifier):
    def _backprop(self, X, y, sample_weight, activations, deltas, coef_grads, intercept_grads):
        # 原样返回库计算的损失和梯度；此时尚未执行本批次权重更新。
        result = super()._backprop(X, y, sample_weight, activations, deltas, coef_grads, intercept_grads)
        if not hasattr(self, 'batch_records'):
            self.batch_records = []
        epoch = self.n_iter_ + 1
        previous = self.batch_records[-1] if self.batch_records else None
        self.batch_records.append({
            'step': len(self.batch_records) + 1, 'epoch': epoch,
            'batch_in_epoch': previous['batch_in_epoch'] + 1 if previous and previous['epoch'] == epoch else 1,
            'batch_size': X.shape[0], 'batch_training_objective': float(result[0]),
        })
        return result

    def _update_no_improvement_count(self, early_stopping, X, y, sample_weight):
        # 先执行原实现；保持训练损失控制的停止计数不变。
        super()._update_no_improvement_count(early_stopping, X, y, sample_weight)
        self.epoch_recorder(self)


def check_hook():
    """小数据校验：插入评分钩子不改变训练轨迹、参数或预测。"""
    expected = ['self', 'early_stopping', 'X', 'y', 'sample_weight']
    assert list(inspect.signature(MLPClassifier._update_no_improvement_count).parameters) == expected
    assert list(inspect.signature(MLPClassifier._backprop).parameters) == [
        'self', 'X', 'y', 'sample_weight', 'activations', 'deltas', 'coef_grads', 'intercept_grads']
    rng = np.random.RandomState(7)
    X = rng.normal(size=(30, 5))
    y = np.arange(30) % 3
    params = dict(hidden_layer_sizes=(4,), max_iter=4, random_state=42, batch_size=7)
    plain, observed = MLPClassifier(**params), ObservedMLP(**params)
    epochs = []
    def record(model):
        probability = model.predict_proba(X)
        log_loss(y, probability, labels=model.classes_)
        epochs.append(model.n_iter_)
    observed.epoch_recorder = record
    with warnings.catch_warnings(), threadpool_limits(limits=THREAD_LIMIT):
        warnings.simplefilter('ignore')
        plain.fit(X, y)
        observed.fit(X, y)
    assert epochs == [1, 2, 3, 4]
    np.testing.assert_array_equal(plain.loss_curve_, observed.loss_curve_)
    batches = pd.DataFrame(observed.batch_records)
    assert len(batches) == 20
    assert batches.batch_size.tolist() == [7, 7, 7, 7, 2] * 4
    for epoch, group in batches.groupby('epoch'):
        np.testing.assert_allclose(np.average(group.batch_training_objective, weights=group.batch_size),
                                   observed.loss_curve_[epoch - 1], rtol=1e-14)
    np.testing.assert_array_equal(plain.predict_proba(X), observed.predict_proba(X))
    for a, b in zip(plain.coefs_ + plain.intercepts_, observed.coefs_ + observed.intercepts_):
        np.testing.assert_array_equal(a, b)


def main():
    args = baseline_arguments(svc=False, mlp=True)
    if args.check_only:
        return
    BASELINE = args.mlp_baseline
    check_hook()
    data, _ = load_data()
    train_texts, val_texts, y_train, y_val = split_data(data)
    vectorizer, X_train, X_val = build_tfidf(train_texts.map(clean_headers), val_texts.map(clean_headers), ngram_range=(1, 1), min_df=1)
    out = EXPERIMENT_RESULTS / datetime.now().strftime('mlp_batch_trace_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    model = ObservedMLP(**FINAL_MLP)
    baseline_meta = json.loads((BASELINE / 'metadata.json').read_text(encoding='utf-8'))
    meta = {
        'status':'running', 'created_at':datetime.now().astimezone().isoformat(),
        'parameters':model.get_params(), 'tfidf_parameters':vectorizer.get_params(),
        'preprocessing':'clean_headers', 'random_state':RANDOM_STATE,
        'python':platform.python_version(), 'sklearn':sklearn.__version__,
        'numpy':np.__version__, 'pandas':pd.__version__, 'thread_limit': THREAD_LIMIT,
        'train_csv_sha256':hashlib.sha256((DATA_DIR / 'train_data.csv').read_bytes()).hexdigest(),
        'baseline':str(BASELINE.resolve()),
        'hook':'ObservedMLP._update_no_improvement_count; super called first; read-only epoch scoring',
        'batch_hook':'ObservedMLP._backprop: call super once, record original returned loss before optimizer update, return unchanged result',
        'batch_loss_definition':'Original sklearn mini-batch mean cross-entropy plus L2; pre-update weights; no smoothing or resampling. Last batch may be smaller.',
        'hook_smoke_test':'passed: identical weights and losses on small data',
        'loss_definition':'train_cross_entropy and val_cross_entropy: mean sklearn.log_loss at epoch end, no L2; training_objective: original batch-aggregated loss_curve_ including L2',
        'stopping':'unchanged training-loss stopping; validation never used to stop or update weights',
        'caveat':'Single previously reused validation split and one seed; best epoch is selected on validation data, not independent test evidence.',
    }
    def save_meta():
        (out / 'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    assert meta['train_csv_sha256'] == baseline_meta['train_csv_sha256']
    assert json.loads(json.dumps(meta['parameters'])) == baseline_meta['parameters']
    assert json.loads(json.dumps(meta['tfidf_parameters'],default=str)) == baseline_meta['tfidf_parameters']
    split = pd.concat([pd.DataFrame({'row_index':train_texts.index,'split':'train'}),pd.DataFrame({'row_index':val_texts.index,'split':'validation'})],ignore_index=True)
    assert split.equals(pd.read_csv(BASELINE / 'split.csv'))
    split.to_csv(out / 'split.csv',index=False)
    save_meta()
    for filename in (Path(__file__).name,'prepare_features.py','compare_header_cleaning.py','train_mlp.py','plot_mlp_batches.py'):
        (out / filename).write_bytes(source_path(filename).read_bytes())
        snapshot_common(out)
    rows, probabilities = [], []
    start = perf_counter()

    def record_epoch(current):
        began = perf_counter()
        train_prob = current.predict_proba(X_train)
        val_prob = current.predict_proba(X_val)
        train_pred = current.classes_[train_prob.argmax(axis=1)]
        val_pred = current.classes_[val_prob.argmax(axis=1)]
        row = {
            'epoch':current.n_iter_, 'training_objective':current.loss_curve_[-1],
            'train_cross_entropy':log_loss(y_train,train_prob,labels=current.classes_),
            'val_cross_entropy':log_loss(y_val,val_prob,labels=current.classes_),
            'train_accuracy':accuracy_score(y_train,train_pred),
            'val_accuracy':accuracy_score(y_val,val_pred),
            'train_macro_f1':f1_score(y_train,train_pred,average='macro'),
            'val_macro_f1':f1_score(y_val,val_pred,average='macro'),
            'evaluation_seconds':perf_counter()-began,
            'elapsed_seconds':perf_counter()-start,
        }
        rows.append(row)
        pd.DataFrame(current.batch_records).to_csv(out / 'batch_metrics.csv', index=False)
        probabilities.append(val_prob.copy())
        pd.DataFrame(rows).to_csv(out / 'epoch_metrics.csv',index=False)
        # 每轮保存最新验证概率，最优交叉熵轮次另存；不改变训练模型。
        np.savez_compressed(out / 'latest_validation.npz',epoch=current.n_iter_,row_index=val_texts.index.to_numpy(),target=y_val.to_numpy(),classes=current.classes_,probabilities=val_prob)
        if row['val_cross_entropy'] == min(r['val_cross_entropy'] for r in rows):
            np.savez_compressed(out / 'best_loss_validation.npz',epoch=current.n_iter_,row_index=val_texts.index.to_numpy(),target=y_val.to_numpy(),classes=current.classes_,probabilities=val_prob)
        print(f"  epoch={current.n_iter_} train_CE={row['train_cross_entropy']:.6f} val_CE={row['val_cross_entropy']:.6f} train_acc={row['train_accuracy']:.4%} val_acc={row['val_accuracy']:.4%}",flush=True)

    model.epoch_recorder = record_epoch
    print(f'记录目录：{out}\n钩子验证通过；原训练算法和停止规则保持不变。',flush=True)
    with threadpool_limits(limits=THREAD_LIMIT), (out / 'training.log').open('w',encoding='utf-8') as log:
        meta['threadpools'] = threadpool_info()
        save_meta()
        with contextlib.redirect_stdout(Tee(sys.stdout,log)), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(X_train,y_train)
    metrics = pd.DataFrame(rows)
    assert len(metrics) == model.n_iter_
    batches = pd.DataFrame(model.batch_records)
    for epoch, group in batches.groupby('epoch'):
        assert group.batch_size.sum() == X_train.shape[0]
        np.testing.assert_allclose(np.average(group.batch_training_objective, weights=group.batch_size),
                                   metrics.iloc[epoch - 1].training_objective, rtol=1e-12, atol=1e-14)
    meta['batch_count'] = len(batches)
    meta['batch_weighted_means_match_epoch_objective'] = True
    np.savez_compressed(out / 'all_validation_probabilities.npz',epochs=metrics.epoch.to_numpy(),row_index=val_texts.index.to_numpy(),target=y_val.to_numpy(),classes=model.classes_,probabilities=np.stack(probabilities))
    for i, probability in enumerate(probabilities):
        assert abs(log_loss(y_val,probability,labels=model.classes_)-metrics.iloc[i].val_cross_entropy)<1e-12
    original = pd.read_csv(BASELINE / 'loss_curve.csv')
    same = len(original)==len(metrics) and np.allclose(original.training_loss,metrics.training_objective,rtol=1e-10,atol=1e-12)
    baseline_predictions = pd.read_csv(BASELINE / 'validation_predictions.csv')
    final_predictions = model.classes_[probabilities[-1].argmax(axis=1)]
    same_predictions = np.array_equal(final_predictions,baseline_predictions.prediction.to_numpy())
    pd.DataFrame({'row_index':val_texts.index,'target':y_val.to_numpy(),'prediction':final_predictions}).to_csv(out / 'final_validation_predictions.csv',index=False)
    meta.update(status='interrupted' if any('interrupted' in str(w.message).lower() for w in caught) else 'complete',finished_at=datetime.now().astimezone().isoformat(),warnings=[str(w.message) for w in caught],baseline_loss_curve_matches=bool(same),baseline_predictions_match=bool(same_predictions),total_seconds=perf_counter()-start)
    save_meta()
    best_loss = metrics.loc[metrics.val_cross_entropy.idxmin()]
    best_accuracy = metrics.sort_values(['val_accuracy','val_macro_f1','epoch'],ascending=[False,False,True]).iloc[0]
    selected = pd.DataFrame([best_loss,best_accuracy,metrics.iloc[-1]],index=['minimum_validation_CE','best_validation_accuracy_then_F1','final'])
    selected.to_csv(out / 'selected_epochs.csv',index_label='criterion')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(metrics.epoch,metrics.train_cross_entropy,label='Train CE')
    axes[0,0].plot(metrics.epoch,metrics.val_cross_entropy,label='Validation CE')
    axes[0,0].set(yscale='log',ylabel='Cross-entropy (no L2)',title='Same metric, same epoch-end model')
    axes[0,0].legend()
    axes[0,1].plot(metrics.epoch,100*metrics.train_accuracy,label='Train')
    axes[0,1].plot(metrics.epoch,100*metrics.val_accuracy,label='Validation')
    axes[0,1].set(ylabel='Accuracy (%)',title='Epoch-end accuracy')
    axes[0,1].legend()
    axes[1,0].plot(metrics.epoch,metrics.val_cross_entropy,label='Validation CE')
    axes[1,0].scatter([best_loss.epoch],[best_loss.val_cross_entropy],label=f'Minimum: epoch {int(best_loss.epoch)}')
    axes[1,0].set(ylabel='Cross-entropy (no L2)',title='Validation loss (linear scale)')
    axes[1,0].legend()
    axes[1,1].plot(metrics.epoch,metrics.training_objective)
    axes[1,1].set(yscale='log',ylabel='Training objective (including L2)',title='Original sklearn loss_curve_')
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(out / 'learning_curves.png',dpi=160)
    plt.close(fig)
    from experiments.plot_mlp_batches import plot_batches
    plot_batches(out)
    lines=['# MLP逐轮记录实验','',
           '保持(100,)、alpha=0.0001、ReLU、Adam、随机种子42、max_iter=300与原训练损失停止规则。',
           '每轮结束后，用该轮同一模型分别计算训练/验证纯交叉熵、Accuracy、Macro-F1。验证指标只记录，不改变权重或停止。',
           f'与原基线训练目标曲线一致：{same}；最终验证预测一致：{same_predictions}。','',
           '| 选择标准 | epoch | 训练CE | 验证CE | 训练Accuracy | 验证Accuracy | 验证Macro-F1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in selected.iterrows():
        lines.append(f'| {name} | {int(r.epoch)} | {r.train_cross_entropy:.6f} | {r.val_cross_entropy:.6f} | {r.train_accuracy:.4%} | {r.val_accuracy:.4%} | {r.val_macro_f1:.6f} |')
    lines += ['', '最低交叉熵轮次与最高准确率轮次是两个不同标准，均在同一验证集上选择，不是独立测试成绩。',
              f'新增 {len(batches)} 条原始小批次损失，保存为 batch_metrics.csv；含L2、更新前计算，未平滑。按批次样本量加权后与每轮原训练目标一致。',
              'training_objective 是轮内各批次训练目标（含L2）的汇总，不等于轮末训练纯交叉熵；两者分开记录。',
              '记录钩子依赖当前sklearn内部接口，升级版本时需重新校验；小数据测试确认不会改变训练参数轨迹。',
              'all_validation_probabilities.npz 保存全部轮次的验证概率、类别顺序、样本索引和标签，可重新核对指标。']
    (out / 'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
