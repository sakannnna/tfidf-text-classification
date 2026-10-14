"""不训练：检查导入、参数、路径、公共函数及历史产物完整性。"""
import ast
import hashlib
import importlib
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from common.config import DATA_DIR, FINAL_RESULTS, EXPERIMENT_RESULTS, AUDIT_RESULTS, FINAL_TFIDF, source_path
from common.config import SOURCE_DIR, PROJECT_DIR
from common.models import make_models
from common.prepare_features import load_data, split_data, build_tfidf
from common.preprocessing import clean_headers

class StructureTests(unittest.TestCase):
    def test_root_entry_and_cli_guard(self):
        import runpy
        import subprocess
        import sys
        with patch.object(MLPClassifier, 'fit', side_effect=AssertionError('禁止训练')), patch.object(SVC, 'fit', side_effect=AssertionError('禁止训练')), patch.object(MultinomialNB, 'fit', side_effect=AssertionError('禁止训练')):
            for entry in PROJECT_DIR.glob('*.py'):
                namespace = runpy.run_path(str(entry))
                tree = ast.parse(entry.read_text(encoding='utf-8'))
                for node in tree.body:
                    if isinstance(node, ast.If):
                        for call in ast.walk(node):
                            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                                self.assertIn(call.func.id, namespace, str(entry))
        runner = PROJECT_DIR / 'run.py'
        result = subprocess.run([sys.executable, '-X', 'utf8', str(runner), 'predict'], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'--execute', result.stderr)

    def test_imports_do_not_train(self):
        with patch.object(MLPClassifier, "fit", side_effect=AssertionError("禁止训练")), patch.object(SVC, "fit", side_effect=AssertionError("禁止训练")), patch.object(MultinomialNB, "fit", side_effect=AssertionError("禁止训练")):
            for folder in ("common", "final", "experiments", "audit"):
                for path in (SOURCE_DIR / folder).glob("*.py"):
                    ast.parse(path.read_text(encoding="utf-8"))
                    importlib.import_module(folder + "." + path.stem)

    def test_paths_and_hook_signature(self):
        expected = ["self", "early_stopping", "X", "y", "sample_weight"]
        self.assertEqual(list(inspect.signature(MLPClassifier._update_no_improvement_count).parameters), expected)
        for name in ("prepare_features.py", "preprocessing.py", "models.py", "logging_utils.py", "train_mlp.py"):
            self.assertTrue(source_path(name).is_file())
    def test_locked_parameters_without_history(self):
        models = make_models()
        mlp = models["mlp"].get_params()
        for key, value in dict(hidden_layer_sizes=(100,), alpha=0.0001, activation="relu",
                               solver="adam", max_iter=300, random_state=42,
                               early_stopping=False, verbose=True).items():
            self.assertEqual(mlp[key], value)
        svc = models["svc"].get_params()
        for key, value in dict(kernel="linear", C=10, random_state=42).items():
            self.assertEqual(svc[key], value)
        self.assertEqual(FINAL_TFIDF, {"ngram_range": (1,1), "min_df": 1})

    def test_tfidf_does_not_learn_validation_words(self):
        vec, train, validation = build_tfidf(["alpha beta", "beta gamma"], ["unseenword alpha"], **FINAL_TFIDF)
        self.assertNotIn("unseenword", vec.vocabulary_)
        self.assertEqual(train.shape[1], validation.shape[1])
        self.assertTrue(vec.sublinear_tf)
        self.assertEqual(vec.norm, "l2")
        self.assertEqual(vec.min_df, 1)

    def test_split_is_reproducible_disjoint_and_stratified(self):
        import pandas as pd
        data = pd.DataFrame({"text": [f"text{i}" for i in range(50)], "target":[i % 2 for i in range(50)]})
        a, b, ya, yb = split_data(data)
        c, d, _, _ = split_data(data)
        self.assertEqual(a.index.tolist(), c.index.tolist())
        self.assertEqual(b.index.tolist(), d.index.tolist())
        self.assertFalse(set(a.index) & set(b.index))
        self.assertEqual(len(a), 40)
        self.assertEqual(len(b), 10)
        self.assertEqual(yb.value_counts().to_dict(), {0:5, 1:5})
