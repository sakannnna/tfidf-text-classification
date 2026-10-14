"""固定线性 SVC C=10，比较已保存的单词基线与新训练的单词+双词模型。"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
from common.baselines import baseline_arguments, validate_baseline, svc_prediction_column
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common.prepare_features import DATA_DIR
from experiments.tune_svc import run_experiment




def main():
    args = baseline_arguments(svc=True, mlp=False)
    if args.check_only:
        return
    BASELINE = args.svc_baseline
    output = run_experiment(kernel="linear", c_values=[10.0], ngram_range=(1, 2))
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    old_meta = json.loads((BASELINE / "metadata.json").read_text(encoding="utf-8"))
    new_meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert old_meta["train_csv_sha256"] == new_meta["train_csv_sha256"]
    assert pd.read_csv(BASELINE / "split.csv").equals(pd.read_csv(output / "split.csv"))
    assert {k:v for k,v in old_meta['tfidf_parameters'].items() if k!='ngram_range'} == {k:v for k,v in new_meta['tfidf_parameters'].items() if k!='ngram_range'}
    old = pd.read_csv(BASELINE / "validation_predictions.csv")
    new = pd.read_csv(output / "validation_predictions.csv")
    assert old[['row_index','target']].equals(new[['row_index','target']])
    compared = old[['row_index','target']].copy()
    compared['unigram_prediction'] = old[svc_prediction_column(old)]
    compared['bigram_prediction'] = new['C_10.0_gamma_scale']
    a = compared.unigram_prediction == compared.target
    b = compared.bigram_prediction == compared.target
    compared['transition'] = ['fixed' if not x and y else 'new_error' if x and not y else 'both_correct' if x else 'both_wrong' for x,y in zip(a,b)]
    compared.to_csv(output / 'prediction_comparison.csv', index=False)
    data = pd.read_csv(DATA_DIR / 'train_data.csv')
    changed = compared[compared.transition.isin(['fixed','new_error'])].copy()
    changed['text'] = data.loc[changed.row_index,'text'].to_numpy()
    changed.to_csv(output / 'changed_errors.csv', index=False)
    metrics = []
    summaries = []
    for name, pred, path in [('unigram', compared.unigram_prediction, BASELINE), ('unigram_bigram', compared.bigram_prediction, output)]:
        report = classification_report(compared.target,pred,output_dict=True,zero_division=0)
        for label in range(10):
            metrics.append({'features':name,'label':label,**report[str(label)]})
        cm = confusion_matrix(compared.target,pred,labels=range(10))
        pd.DataFrame(cm,index=pd.Index(range(10),name='true_label'),columns=range(10)).to_csv(output / f'{name}_confusion_matrix.csv')
        saved = pd.read_csv(path / 'results.csv')
        saved = saved.loc[saved.C == 10.0].iloc[0]
        summaries.append({'features':name,'accuracy':accuracy_score(compared.target,pred),'macro_f1':f1_score(compared.target,pred,average='macro'),'vocabulary_size':int(saved.vocabulary_size),'fit_seconds':saved.fit_seconds,'predict_seconds':saved.predict_seconds})
    pd.DataFrame(metrics).to_csv(output / 'comparison_per_class.csv',index=False)
    pd.DataFrame(summaries).to_csv(output / 'feature_comparison.csv',index=False)
    counts = compared.transition.value_counts().to_dict()
    lines = ['# 线性 SVC 的短语特征对照','',f'固定 C=10、min_df=1、邮件头清洗和分层划分。单词基线复用 {BASELINE.name} 的记录；已核对数据指纹、划分及其他 TF-IDF 参数。','',
             '| 特征 | Accuracy | Macro-F1 | 词表 | 训练秒数 | 预测秒数 |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['features']} | {r['accuracy']:.4%} | {r['macro_f1']:.6f} | {r['vocabulary_size']} | {r['fit_seconds']:.3f} | {r['predict_seconds']:.3f} |")
    lines += ['',f'错误迁移：{counts}','', '## 易混淆类别 F1','', '| 标签 | 单词 | 单词+双词 |','|---:|---:|---:|']
    for label in [0,1,2,7]:
        values = [r['f1-score'] for r in metrics if r['label']==label]
        lines.append(f'| {label} | {values[0]:.6f} | {values[1]:.6f} |')
    for name,pred in [('单词',compared.unigram_prediction),('单词+双词',compared.bigram_prediction)]:
        count = ((pred != compared.target) & pred.isin([0,1,2,7]) & compared.target.isin([0,1,2,7])).sum()
        lines += ['', f'{name}在0/1/2/7内部的混淆：{count}条。']
    lines += ['', '单次已用于调参的验证划分，不能视为独立测试结论。历史与本次耗时仅供参考，未做重复计时。']
    (output / 'feature_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
