"""小网格：隐藏层50/100 × alpha=0.0001/0.001/0.01。

复用已有100、0.0001基线，新增5次训练；每组保留完整运行记录。
python tune_mlp.py --resume results/某次目录 可继续未完成的组合。
"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import argparse
from common.baselines import baseline_arguments, validate_baseline, svc_prediction_column
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common.prepare_features import DATA_DIR
from common.config import SOURCE_DIR
from experiments.train_mlp import run_mlp


CONFIGURATIONS = [(100, 0.0001), (100, 0.001), (100, 0.01), (50, 0.0001), (50, 0.001), (50, 0.01)]


def summarize(root, runs, BASELINE):
    """校验每组预测与设置，增量更新汇总，保留全部配置。"""
    baseline_meta = json.loads((BASELINE / 'metadata.json').read_text(encoding='utf-8'))
    baseline_split = pd.read_csv(BASELINE / 'split.csv')
    base_predictions = pd.read_csv(BASELINE / 'validation_predictions.csv')
    rows = []
    loss_tables = []
    for width, alpha, path in runs:
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        assert meta['status'] == 'complete'
        assert meta['train_csv_sha256'] == baseline_meta['train_csv_sha256']
        assert meta['tfidf_parameters'] == baseline_meta['tfidf_parameters']
        assert meta['preprocessing'] == baseline_meta['preprocessing']
        for key, value in baseline_meta['parameters'].items():
            if key not in ('hidden_layer_sizes','alpha'):
                assert meta['parameters'][key] == value, key
        assert meta['parameters']['hidden_layer_sizes'] == [width]
        assert meta['parameters']['alpha'] == alpha
        assert pd.read_csv(path / 'split.csv').equals(baseline_split)
        pred = pd.read_csv(path / 'validation_predictions.csv')
        assert pred[['row_index','target']].equals(base_predictions[['row_index','target']])
        record = pd.read_csv(path / 'results.csv').iloc[0].to_dict()
        assert abs(accuracy_score(pred.target,pred.prediction)-record['accuracy']) < 1e-12
        assert abs(f1_score(pred.target,pred.prediction,average='macro')-record['macro_f1']) < 1e-12
        curve = pd.read_csv(path / 'loss_curve.csv')
        assert len(curve) == record['epochs'] and np.isfinite(curve.training_loss).all()
        loss_tables.append(curve.assign(hidden_units=width, alpha=alpha))
        rows.append({'hidden_units':width,'alpha':alpha,**record,
                     'generalization_gap_pp':100*(record['train_accuracy']-record['accuracy']),
                     'source_run':str(path.resolve()), 'reused_baseline':path == BASELINE})
    frame = pd.DataFrame(rows).sort_values(['accuracy','macro_f1'], ascending=False, kind='stable')
    frame.to_csv(root / 'results.csv', index=False)
    pd.concat(loss_tables,ignore_index=True).to_csv(root / 'all_loss_curves.csv',index=False)
    lines = ['# MLP 网络大小与正则化对照', '',
             '固定随机种子42、原80/20划分、邮件头清洗、单词TF-IDF（min_df=1）、ReLU、Adam和最多300轮。',
             '停止条件仍是训练损失改善，无外部验证集早停。已有100神经元、alpha=0.0001基线复用历史记录。', '',
             f'已完成 {len(rows)}/6 组。按验证Accuracy、Macro-F1降序排列。', '',
             '| 隐藏神经元 | alpha | 验证Accuracy | Macro-F1 | 训练Accuracy | 训练−验证（百分点） | 轮数 | 训练秒数 | 参数量 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in frame.itertuples(index=False):
        lines.append(f'| {r.hidden_units} | {r.alpha} | {r.accuracy:.4%} | {r.macro_f1:.6f} | {r.train_accuracy:.4%} | {r.generalization_gap_pp:.4f} | {int(r.epochs)} | {r.fit_seconds:.2f} | {int(r.parameter_count)} |')
    lines += ['', '每组原始预测、配置、代码、训练日志、逐轮损失和线性/对数损失图见对应子目录；复用基线来源见results.csv。',
              'log_loss_comparison.png 使用原始epoch损失，不做平滑。不同alpha的损失包含不同强度的L2项，不能只按最终损失大小挑选模型。',
              '单一划分、单一初始化种子，属于验证集调参；没有验证不同初始化下的稳定性。耗时包含不同时间段的历史运行，仅供参考。']
    (root / 'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,width in zip(axes,[100,50]):
        for table in loss_tables:
            if table.hidden_units.iloc[0] == width:
                ax.plot(table.epoch,table.training_loss,label=f"alpha={table.alpha.iloc[0]:g}")
        ax.set(xlabel='Epoch',yscale='log',title=f'{width} hidden units')
        if ax.lines:
            ax.legend()
    axes[0].set_ylabel('Training objective including L2 (log scale)')
    fig.tight_layout()
    fig.savefig(root / 'log_loss_comparison.png',dpi=160)
    plt.close(fig)
    return frame


def main():
    args = baseline_arguments(svc=True, mlp=True, resume=True)
    if args.check_only:
        return
    BASELINE = args.mlp_baseline
    root = args.resume.resolve() if args.resume else EXPERIMENT_RESULTS / datetime.now().strftime('mlp_grid_%Y%m%d_%H%M%S_%f')
    if not args.resume:
        root.mkdir(parents=True,exist_ok=False)
        for filename in ('tune_mlp.py','train_mlp.py','prepare_features.py','compare_header_cleaning.py'):
            (root / filename).write_bytes(source_path(filename).read_bytes())
            snapshot_common(root)
    else:
        # 结构调整后不允许以新公共代码续跑旧快照，避免混合实现。
        for current in (SOURCE_DIR / 'common').glob('*.py'):
            saved = root / 'common' / current.name
            assert saved.is_file() and saved.read_bytes() == current.read_bytes(), '公共模块变化或旧快照缺失，请建立新实验'
        for filename in ('tune_mlp.py','train_mlp.py','prepare_features.py','compare_header_cleaning.py'):
            assert (root / filename).read_bytes() == source_path(filename).read_bytes(), '续跑前代码发生变化，请建立新实验'
    print(f'网格记录目录：{root}',flush=True)
    manifest = {'status':'running','configurations':CONFIGURATIONS,'baseline':str(BASELINE.resolve()), 'runs':[]}
    runs = []
    for width,alpha in CONFIGURATIONS:
        path = BASELINE if (width,alpha)==(100,0.0001) else root / f'hidden_{width}_alpha_{alpha}'
        complete = (path / 'metadata.json').exists() and json.loads((path / 'metadata.json').read_text(encoding='utf-8')).get('status') == 'complete'
        if not complete:
            if path.exists():
                # 保留中断运行的日志，重新训练时使用新子目录。
                path = root / f'hidden_{width}_alpha_{alpha}_retry_{datetime.now():%Y%m%d_%H%M%S_%f}'
            run_mlp(hidden_layer_sizes=(width,),alpha=alpha,output_dir=path,svc_baseline=args.svc_baseline)
        runs.append((width,alpha,path))
        manifest['runs'].append({'hidden_units':width,'alpha':alpha,'path':str(path.resolve())})
        frame = summarize(root,runs,BASELINE)
        (root / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'汇总已保存：{len(runs)}/6组',flush=True)
    manifest['status']='complete'
    (root / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(frame[['hidden_units','alpha','accuracy','macro_f1','epochs','fit_seconds']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
