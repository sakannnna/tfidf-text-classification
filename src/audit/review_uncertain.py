"""只为语义复核检索训练例子，不修改预测，不产生真实标签。"""
from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import json
from pathlib import Path
import joblib
import pandas as pd
from common.preprocessing import clean_headers

BASE = DATA_DIR
RUN = FINAL_RESULTS / 'test_predictions_20260921_231958_022008'

def evidence():
    rows = []
    for name in ['spotcheck_annotations.json', 'remaining45_annotations.json', 'agreement100_annotations.json']:
        rows.extend(dict(x, source=name) for x in json.loads((AUDIT_RESULTS/'test_predictions_20260921_231958_022008'/name).read_text(encoding='utf-8'))['annotations'] if x['label'] is None)
    train = pd.read_csv(BASE/'train_data.csv')
    test = pd.read_csv(BASE/'test_data_unlabeled.csv')
    vec = joblib.load(RUN/'tfidf.joblib')
    texts = train.text.map(clean_headers)
    matrix = vec.transform(texts)
    result = []
    for item in rows:
        text = clean_headers(test.iloc[item['test_row_index']].text)
        scores = (vec.transform([text]) @ matrix.T).toarray()[0]
        nearest = scores.argsort()[-3:][::-1]
        item.update(text=text, neighbors=[{'train_row_index':int(i), 'label':int(train.iloc[i].target), 'cosine':round(float(scores[i]),4), 'text':texts.iloc[i]} for i in nearest])
        result.append(item)
    return result

if __name__ == '__main__':
    for x in evidence():
        print('\nID', x['test_row_index'], 'LEN',len(x['text']), '\n',x['text'][:1050])
        for n in x['neighbors']:
            print('NEIGHBOR',n['train_row_index'],n['label'],n['cosine'],repr(n['text'][:260]))
