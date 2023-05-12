from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class HeadHook(Hook):

    def __init__(self):
        pass

    def before_run(self, runner):

        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        runner.model.module.semantic_head.max_iter = len(runner.data_loader)
        runner.model.module.bbox_head.loss_cls.max_iter = len(runner.data_loader)
        runner.model.module.mask_head.loss_mask.max_iter = len(runner.data_loader)
        runner.model.module.semantic_head.epoch = runner.epoch
        runner.model.module.bbox_head.loss_cls.epoch = runner.epoch
        runner.model.module.mask_head.loss_mask.epoch = runner.epoch


        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        runner.model.module.semantic_head.iter = runner.inner_iter
        runner.model.module.bbox_head.loss_cls.iter = runner.iter
        runner.model.module.mask_head.loss_mask.iter = runner.iter


        pass

    def after_iter(self, runner):
        pass