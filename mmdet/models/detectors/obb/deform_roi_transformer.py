
from mmdet.models.builder import DETECTORS
from .deform_obb_two_stage import DeformOBBTwoStageDetector


@DETECTORS.register_module()
class DeformRoITransformer(DeformOBBTwoStageDetector):

    def __init__(self,
                 backbone,
                 deform=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DeformRoITransformer, self).__init__(
            backbone=backbone,
            deform=deform,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
