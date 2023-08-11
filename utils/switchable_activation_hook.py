from up.utils.general.registry_factory import HOOK_REGISTRY
from up.utils.general.hook_helper import Hook
from up.utils.env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger
from ..models.backbones.switchable_activations.utils_ds import Learnable_Relu, Learnable_Relu6
from torch import nn

__all__ = ['SwitchableActivationHook']


def get_target_func(name: str):
    func = {'Learnable_Relu': Learnable_Relu,
            'Learnable_Relu6': Learnable_Relu6,
            'Identity': nn.Identity}
    return func[name]


def replace_activation(net: nn.Module, target_func: nn.Module = nn.Identity, target_args={}, prefix='', keywords=None):
    params = []
    for key, item in net.named_children():
        if isinstance(item, (nn.ReLU, nn.ReLU6, nn.GELU)):
            full_key = (prefix + '.' + key)[1:]
            logger.info('replace hook: %s' % full_key)
            keyword_match = False
            if keywords is None:
                keyword_match = True
            else:
                for keyword in keywords:
                    if keyword in full_key:
                        logger.info(f'replace hook: match {full_key} vs {keyword}')
                        keyword_match = True

            if keyword_match:
                if target_func == nn.Identity:
                    net.__setattr__(key, target_func())
                    net.__getattr__(key).cuda()
                else:
                    net.__setattr__(key, target_func(slope_init=target_args['slope_init']))
                    net.__getattr__(key).cuda()
                    params += [net.__getattr__(key).slope_param]
        else:
            params += replace_activation(item,
                                         target_func, target_args, prefix + '.' + key, keywords)
    return params


@HOOK_REGISTRY.register('switchable_activation')
class SwitchableActivationHook(Hook):
    def __init__(self, runner, activation_cfg, ):
        super(SwitchableActivationHook, self).__init__(runner)
        self.ema_init = False
        self.activation_cfg = activation_cfg
        state_dict_keys_prev = self.runner_ref().model.state_dict().keys()
        new_params = replace_activation(self.runner_ref().model,
                                        get_target_func(self.activation_cfg['type']),
                                        self.activation_cfg['kwargs'],
                                        '',
                                        self.activation_cfg.get('keywords', None))
        self.runner_ref().optimizer.param_groups[0]['params'].extend(new_params)
        state_dict_keys_aft = self.runner_ref().model.state_dict().keys()
        for key in state_dict_keys_aft:
            if key not in state_dict_keys_prev:
                logger.info('ADD key: %s' % (key))
        # print(self.runner_ref().model)

    def before_train(self):
        logger.info('act hook before train')
        logger.info(str(self.runner_ref().ema))
        logger.info(str(self.runner_ref().ema.model))
        if not self.ema_init and self.runner_ref().ema is not None:
            replace_activation(self.runner_ref().ema.model,
                               get_target_func(self.activation_cfg['type']),
                               self.activation_cfg['kwargs'],
                               '',
                               self.activation_cfg.get('keywords', None))
            self.ema_init = True

    def before_forward(self, cur_iter, input):
        slopes = []
        for key, param in self.runner_ref().model.named_parameters(recurse=True):
            # print(key)
            if 'slope' in key:
                slopes.append(param)
        input['slopes'] = slopes
        return input

# @HOOK_REGISTRY.register('dbb_deploy')
# class DBBDeployHook(Hook):
#     def __init__(self, runner, ):
#         super(DBBDeployHook, self).__init__(runner)
#
#     def before_forward(self, cur_iter, input):
#         slopes = []
#         for key, param in self.runner_ref().model.named_parameters(recurse=True):
#             # print(key)
#             if 'slope' in key:
#                 slopes.append(param)
#         input['slopes'] = slopes
#         return input
