"""统一入口；默认不运行训练，训练任务必须显式添加 --execute。"""
import argparse
import runpy
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent / "src"
TASKS = {
    "predict": "final.predict_test",
    "nb-baseline": "experiments.naive_bayes_baseline",
    "inspect-data": "experiments.inspect_data",
    "compare-cleaning": "experiments.compare_header_cleaning",
    "tune-nb": "experiments.tune_naive_bayes",
    "tune-svc": "experiments.tune_svc",
    "tune-rbf": "experiments.tune_rbf_svc",
    "compare-ngrams": "experiments.compare_svc_ngrams",
    "stability": "experiments.check_stability",
    "mlp-baseline": "experiments.train_mlp",
    "tune-mlp": "experiments.tune_mlp",
    "mlp-depth": "experiments.compare_mlp_depth",
    "mlp-seeds": "experiments.check_mlp_seeds",
    "mlp-epochs": "experiments.record_mlp_epochs",
    "errors": "audit.analyze_errors",
    "review-uncertain": "audit.review_uncertain",
}
READ_ONLY = {"inspect-data", "review-uncertain"}
BASELINE_TASKS = {"mlp-baseline", "tune-mlp", "mlp-depth", "mlp-seeds", "mlp-epochs", "compare-ngrams"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["check", *TASKS], nargs="?")
    parser.add_argument("--execute", action="store_true", help="允许所选任务执行训练或写入新的分析结果")
    args, extra = parser.parse_known_args()
    if args.task is None:
        parser.print_help()
        return
    sys.path.insert(0, str(CODE_DIR))
    if args.task == "check":
        if extra:
            parser.error("check 不接受附加参数")
        import unittest
        pattern = "test_*.py"
        suite = unittest.defaultTestLoader.discover(str(CODE_DIR / "tests"), pattern=pattern, top_level_dir=str(CODE_DIR))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    check_only = args.task in BASELINE_TASKS and "--check-only" in extra
    if args.task not in READ_ONLY and not args.execute and not check_only:
        parser.error("此任务可能训练模型或写入结果；确认后添加 --execute。仅检查请运行 check。")
    sys.argv = [TASKS[args.task], *extra]
    runpy.run_module(TASKS[args.task], run_name="__main__")

if __name__ == "__main__":
    main()
