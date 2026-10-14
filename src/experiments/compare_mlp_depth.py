"""单层(100,)与两层(100,50)对照，其他参数保持不变。"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
from common.baselines import baseline_arguments, validate_baseline, svc_prediction_column
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common.prepare_features import DATA_DIR
from experiments.train_mlp import run_mlp




def main():
    args = baseline_arguments(svc=True, mlp=True)
    if args.check_only:
        return
    BASELINE = args.mlp_baseline
    root = EXPERIMENT_RESULTS / datetime.now().strftime('mlp_depth_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    out = run_mlp(hidden_layer_sizes=(100, 50), alpha=0.0001, output_dir=root / 'hidden_100_50', svc_baseline=args.svc_baseline)
    old_meta = json.loads((BASELINE / 'metadata.json').read_text(encoding='utf-8'))
    new_meta = json.loads((out / 'metadata.json').read_text(encoding='utf-8'))
    for key in ('train_csv_sha256','tfidf_parameters','preprocessing','random_state','thread_limit'):
        assert old_meta[key] == new_meta[key], key
    for key in old_meta['parameters']:
        if key != 'hidden_layer_sizes':
            assert old_meta['parameters'][key] == new_meta['parameters'][key], key
    assert pd.read_csv(BASELINE / 'split.csv').equals(pd.read_csv(out / 'split.csv'))
    one = pd.read_csv(BASELINE / 'validation_predictions.csv')
    two = pd.read_csv(out / 'validation_predictions.csv')
    assert one[['row_index','target']].equals(two[['row_index','target']])
    comparison = one.rename(columns={'prediction':'single_layer'})
    comparison['two_layers'] = two.prediction
    a = comparison.single_layer == comparison.target
    b = comparison.two_layers == comparison.target
    comparison['transition'] = ['fixed' if not x and y else 'new_error' if x and not y else 'both_correct' if x else 'both_wrong' for x,y in zip(a,b)]
    comparison.to_csv(root / 'prediction_comparison.csv',index=False)
    records = []
    for architecture, path, pred in [('(100,)',BASELINE,one),('(100, 50)',out,two)]:
        row = pd.read_csv(path / 'results.csv').iloc[0].to_dict()
        assert abs(accuracy_score(pred.target,pred.prediction)-row['accuracy']) < 1e-12
        assert abs(f1_score(pred.target,pred.prediction,average='macro')-row['macro_f1']) < 1e-12
        assert len(pd.read_csv(path / 'loss_curve.csv')) == row['epochs']
        records.append({'architecture':architecture,**row,'gap_pp':100*(row['train_accuracy']-row['accuracy']),'source_run':str(path.resolve())})
    pd.DataFrame(records).to_csv(root / 'depth_comparison.csv',index=False)
    lines = ['# MLP 深度对照', '',
             '固定alpha=0.0001、ReLU、Adam、max_iter=300、训练损失停止规则、随机种子42和原80/20分层划分。',
             '相同邮件头清洗和单词TF-IDF；单层复用历史基线，已核对数据指纹、划分、特征与其余参数。', '',
             '| 隐藏层 | 验证Accuracy | Macro-F1 | 训练Accuracy | 训练−验证（百分点） | 轮数 | 训练秒数 | 参数量 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['architecture']} | {r['accuracy']:.4%} | {r['macro_f1']:.6f} | {r['train_accuracy']:.4%} | {r['gap_pp']:.4f} | {int(r['epochs'])} | {r['fit_seconds']:.3f} | {int(r['parameter_count'])} |")
    counts = comparison.transition.value_counts().to_dict()
    lines += ['',f'预测迁移：{counts}', '',
              '仅单一随机种子与已用于调参的验证集；改变架构也会改变初始化及优化轨迹。不能据此宣布统计显著优劣，也不能将结果归因于训练轮数。',
              '损失图仅为训练目标（含L2），没有逐轮验证曲线，无法定位何时开始过拟合。历史与当前耗时受运行负载影响，仅供参考。']
    (root / 'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print(f'对照记录：{root}',flush=True)


if __name__ == '__main__':
    main()
