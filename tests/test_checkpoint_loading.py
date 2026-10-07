from sample_factory.algo.learning.learner import Learner

from train_arpe import _patch_checkpoint_loading


def test_checkpoint_loading_patch_is_idempotent():
    _patch_checkpoint_loading()
    first = Learner.load_checkpoint
    _patch_checkpoint_loading()
    assert Learner.load_checkpoint is first
