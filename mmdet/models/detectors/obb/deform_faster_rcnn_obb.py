from mmdet.models.builder import DETECTORS
from .deform_obb_two_stage import DeformOBBTwoStageDetector


@DETECTORS.register_module()
class DeformFasterRCNNOBB(DeformOBBTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 deform,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(DeformFasterRCNNOBB, self).__init__(
            backbone=backbone,
            deform=deform,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
