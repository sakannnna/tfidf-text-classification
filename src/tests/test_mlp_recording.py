"""极小合成数据验证记录钩子，不运行正式实验。"""
import unittest
from experiments.record_mlp_epochs import check_hook


class RecordingTests(unittest.TestCase):
    def test_batch_and_epoch_hooks_preserve_training(self):
        check_hook()
