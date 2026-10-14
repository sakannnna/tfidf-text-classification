"""使用临时合成基线测试接口，不读取历史结果、不训练分类器。"""
import hashlib
import json
import platform
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from common.baselines import validate_baseline, normalized, svc_prediction_column
from common.models import make_models
from common.prepare_features import split_data, build_tfidf
from common.config import FINAL_TFIDF, THREAD_LIMIT

class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = pd.DataFrame({"text":[f"sample {i}" for i in range(50)], "target":[i%2 for i in range(50)]})
        self.data.to_csv(self.root/"train_data.csv", index=False)
        self.base = self.root/"fixture"
        self.base.mkdir()
        tr, va, _, y = split_data(self.data)
        pd.concat([pd.DataFrame({"row_index":tr.index,"split":"train"}),pd.DataFrame({"row_index":va.index,"split":"validation"})],ignore_index=True).to_csv(self.base/"split.csv",index=False)
        self.pred = pd.DataFrame({"row_index":va.index,"target":y.to_numpy(),"C_10.0":y.to_numpy()})
        self.pred.to_csv(self.base/"validation_predictions.csv",index=False)
        vec,_,_ = build_tfidf(["alpha beta"],["alpha"],**FINAL_TFIDF)
        self.meta = {
            "train_csv_sha256":hashlib.sha256((self.root/"train_data.csv").read_bytes()).hexdigest(),
            "preprocessing":"clean_headers","random_state":42,"validation_size":0.2,
            "tfidf_parameters":normalized(vec.get_params()),
            "svc_fixed_parameters":{k:v for k,v in normalized(make_models()["svc"].get_params()).items() if k not in ("C","gamma")},
            "C_values":[10.0], "kernel":"linear","gamma_values":["scale"],
        }
        self.write_meta()
        pd.DataFrame([{"C":10.0,"fit_status":0,"accuracy":1.0,"macro_f1":1.0}]).to_csv(self.base/"results.csv",index=False)

    def write_meta(self):
        (self.base/"metadata.json").write_text(json.dumps(self.meta),encoding="utf-8")

    def check(self, kind="svc"):
        with patch.object(MLPClassifier,"fit",side_effect=AssertionError("训练被禁止")),patch.object(SVC,"fit",side_effect=AssertionError("训练被禁止")):
            return validate_baseline(self.base,kind,data_dir=self.root)

    def test_legacy_and_new_svc_prediction_names(self):
        self.assertEqual(self.check(),self.base.resolve())
        self.pred.rename(columns={"C_10.0":"C_10.0_gamma_scale"}).to_csv(self.base/"validation_predictions.csv",index=False)
        self.assertEqual(self.check(),self.base.resolve())

    def test_missing_file_and_wrong_config_fail(self):
        with self.assertRaisesRegex(ValueError,"缺少文件"):
            validate_baseline(self.root/"missing","svc",data_dir=self.root)
        self.meta["tfidf_parameters"]["min_df"]=2
        self.write_meta()
        with self.assertRaisesRegex(ValueError,"TF-IDF"):
            self.check()

    def test_wrong_hash_split_and_score_fail(self):
        self.meta["train_csv_sha256"]="incorrect"
        self.write_meta()
        with self.assertRaisesRegex(ValueError,"指纹"):
            self.check()
        self.meta["train_csv_sha256"]=hashlib.sha256((self.root/"train_data.csv").read_bytes()).hexdigest()
        self.write_meta()
        split=pd.read_csv(self.base/"split.csv")
        split.iloc[::-1].to_csv(self.base/"split.csv",index=False)
        with self.assertRaisesRegex(ValueError,"划分"):
            self.check()
        split.to_csv(self.base/"split.csv",index=False)
        self.pred["C_10.0"]=1-self.pred.target
        self.pred.to_csv(self.base/"validation_predictions.csv",index=False)
        with self.assertRaisesRegex(ValueError,"准确率"):
            self.check()

    def test_mlp_and_incomplete_run(self):
        self.meta.update(model="MLPClassifier",status="complete",parameters=normalized(make_models()["mlp"].get_params()),
                         python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,thread_limit=THREAD_LIMIT)
        self.write_meta()
        self.pred.rename(columns={"C_10.0":"prediction"}).to_csv(self.base/"validation_predictions.csv",index=False)
        pd.DataFrame([dict(epochs=1,accuracy=1.0,macro_f1=1.0,train_accuracy=1.0,fit_seconds=0,parameter_count=0,warnings="")]).to_csv(self.base/"results.csv",index=False)
        pd.DataFrame({"training_loss":[0.1]}).to_csv(self.base/"loss_curve.csv",index=False)
        self.check("mlp")
        self.meta["status"]="running"
        self.write_meta()
        with self.assertRaisesRegex(ValueError,"未完成"):
            self.check("mlp")

    def test_check_only_dispatch_does_not_train(self):
        import importlib, sys
        tasks = [("train_mlp",True,False),("compare_svc_ngrams",True,False),
                 ("compare_mlp_depth",True,True),("check_mlp_seeds",True,True),
                 ("record_mlp_epochs",False,True),("tune_mlp",True,True)]
        for name,svc,mlp in tasks:
            argv=[name,"--check-only"]
            if svc: argv+=["--svc-baseline",str(self.base)]
            if mlp: argv+=["--mlp-baseline",str(self.base)]
            module=importlib.import_module("experiments."+name)
            with patch.object(sys,"argv",argv),patch("common.baselines.validate_baseline",return_value=self.base),patch.object(MLPClassifier,"fit",side_effect=AssertionError("训练被禁止")),patch.object(SVC,"fit",side_effect=AssertionError("训练被禁止")),patch("pathlib.Path.mkdir",side_effect=AssertionError("不应创建结果目录")):
                module.main()
