from up.utils.general.registry_factory import HOOK_REGISTRY
from up.utils.general.hook_helper import Hook
from up.utils.env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger
from ..models.backbones.switchable_activations.utils_ds import Learnable_Relu, Learnable_Relu6
from torch import nn

__all__ = ['SegLossWeightHook']


@HOOK_REGISTRY.register('seg_loss_weight')
class SegLossWeightHook(Hook):
    def __init__(self, runner, weight, max_epoch):
        super(SegLossWeightHook, self).__init__(runner)
        # self.weight = weight
        self.n_class = len(weight)
        avg = sum(weight) / self.n_class
        self.weight = [i / avg for i in weight]
        self.runner_ref().model.decoder.loss.update_weight([1. for i in range(self.n_class)])
        self.max_epoch = max_epoch

    def after_epoch(self, cur_epoch):
        cur_epoch = min(self.max_epoch, cur_epoch)
        new_weight = [1 + (i - 1) / self.max_epoch * cur_epoch for i in self.weight]
        # avg = sum(new_weight) / self.n_class
        # normed_new_weight = [i / avg for i in new_weight]
        print(new_weight)
        self.runner_ref().model.decoder.loss.update_weight(new_weight)
