"""固定数据划分和超参数，仅重复MLP训练种子42/43/44；42复用基线。"""
from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
from common.baselines import baseline_arguments, validate_baseline, svc_prediction_column
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common.prepare_features import DATA_DIR
from experiments.train_mlp import run_mlp



def main():
    args = baseline_arguments(svc=True, mlp=True)
    if args.check_only:
        return
    BASELINE = args.mlp_baseline
    SVC = args.svc_baseline
    out = EXPERIMENT_RESULTS / datetime.now().strftime('mlp_seeds_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    baseline_meta = json.loads((BASELINE / 'metadata.json').read_text(encoding='utf-8'))
    baseline_split = pd.read_csv(BASELINE / 'split.csv')
    baseline_pred = pd.read_csv(BASELINE / 'validation_predictions.csv')
    svc_pred = pd.read_csv(SVC / 'validation_predictions.csv')
    assert baseline_pred[['row_index','target']].equals(svc_pred[['row_index','target']])
    svc_accuracy = accuracy_score(svc_pred.target, svc_pred[svc_prediction_column(svc_pred)])
    manifest = {'status':'running', 'seeds':[42,43,44], 'split_seed':42,
                'selection':'Fixed (100,), alpha=0.0001; no seed selection; final planned seed remains 42',
                'baseline':str(BASELINE.resolve()), 'runs':[]}
    def save_manifest():
        (out / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save_manifest()
    rows = []
    try:
        for seed in manifest['seeds']:
            path = BASELINE if seed == 42 else run_mlp(output_dir=out / f'seed_{seed}', model_seed=seed, svc_baseline=args.svc_baseline)
            meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
            assert meta['status'] == 'complete'
            expected = dict(baseline_meta['parameters'], random_state=seed)
            assert meta['parameters'] == expected
            for key in ['tfidf_parameters','train_csv_sha256','preprocessing','thread_limit','python','sklearn','numpy','pandas']:
                assert meta[key] == baseline_meta[key], key
            assert pd.read_csv(path / 'split.csv').equals(baseline_split)
            pred = pd.read_csv(path / 'validation_predictions.csv')
            assert pred[['row_index','target']].equals(baseline_pred[['row_index','target']])
            row = pd.read_csv(path / 'results.csv').iloc[0].to_dict()
            assert pd.isna(row['warnings']) or not row['warnings']
            assert abs(accuracy_score(pred.target,pred.prediction)-row['accuracy']) < 1e-12
            assert abs(f1_score(pred.target,pred.prediction,average='macro')-row['macro_f1']) < 1e-12
            assert len(pd.read_csv(path / 'loss_curve.csv')) == row['epochs']
            row.update(model_seed=seed, split_seed=42, source_run=str(path.resolve()),
                       reused=seed == 42, svc_accuracy=svc_accuracy,
                       delta_vs_svc_pp=100*(row['accuracy']-svc_accuracy))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / 'results.csv',index=False)
            manifest['runs'].append({'seed':seed,'source_run':row['source_run'],'verified':True})
            save_manifest()
    except BaseException as error:
        manifest.update(status='interrupted_or_failed',error=repr(error))
        save_manifest()
        raise
    frame = pd.DataFrame(rows)
    summary = frame[['accuracy','macro_f1','train_accuracy','epochs']].agg(['mean','std','min','max'])
    summary.to_csv(out / 'statistics.csv',index_label='statistic')
    lines = ['# MLP训练随机性小规模重复','',
             '固定原80/20分层划分（划分种子42）、邮件头清洗、单词TF-IDF min_df=1、单层(100,)、alpha=0.0001、ReLU/Adam、max_iter=300及原训练损失停止规则。',
             '仅改变模型种子42/43/44，影响权重初始化和训练样本打乱。42复用历史基线；各组数据指纹、划分、参数、软件版本已核对，预测已复算评分。','',
             '| 模型种子 | 验证Accuracy | Macro-F1 | 训练Accuracy | 轮数 | 训练秒数 | 相对线性SVC（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model_seed']} | {r['accuracy']:.4%} | {r['macro_f1']:.6f} | {r['train_accuracy']:.4%} | {int(r['epochs'])} | {r['fit_seconds']:.2f} | {r['delta_vs_svc_pp']:+.4f} |")
    lines += ['',f"Accuracy均值：{frame.accuracy.mean():.4%}；样本标准差：{100*frame.accuracy.std(ddof=1):.4f}个百分点；范围：{frame.accuracy.min():.4%}—{frame.accuracy.max():.4%}。",
              f"Macro-F1均值：{frame.macro_f1.mean():.6f}；样本标准差：{frame.macro_f1.std(ddof=1):.6f}。",
              f"同一划分线性SVC C=10：{svc_accuracy:.4%}；MLP在{int((frame.accuracy>svc_accuracy).sum())}/3个种子下更高。",'',
              '这里只检查固定划分上的训练随机性，不是划分稳定性或独立测试；三次不能证明统计显著优越。报告均值与波动，不挑最高分种子。最终训练种子仍预先保留42，未生成提交预测。',
              '停止规则相同但实际轮数可不同。标准差ddof=1，不是置信区间。历史与本次计时负载不同，不作为严格速度基准。',
              '各组目录保存配置、版本、代码快照、划分、逐条预测、训练日志、损失曲线及分类报告；未保存模型或优化器检查点。']
    (out / 'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest['status']='complete'
    save_manifest()
    print('\n'.join(lines),flush=True)
    print(f'汇总目录：{out}',flush=True)


if __name__ == '__main__':
    main()
