from typing import Dict, List

import torch
from torch import Tensor

from mmcv import ConfigDict
from mmdet.core import bbox2result
from mmdet.models import SingleStageDetector, DETECTORS

from models import build_model
from util.misc import NestedTensor


@DETECTORS.register_module()
class GraftingConditionalDETR(SingleStageDetector):

    def __init__(self, **kwargs):
        super(SingleStageDetector, self).__init__(None)
        self.build_model(ConfigDict(kwargs), self.DEFAULT_ARGS)

    def build_model(self, args: ConfigDict, default_args=None):
        _args = default_args.copy() \
            if default_args is not None else ConfigDict()
        _args.update(args)
        model, criterion, postprocessors = build_model(
            _args, _args.get('num_classes', None))
        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessors['bbox']

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        samples = self.prepare_samples(img, img_metas)
        targets = self.prepare_targets(
            img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = {k: loss * weight_dict[k] for k, loss in loss_dict.items()
                  if k in weight_dict}
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        samples = self.prepare_samples(img, img_metas)
        outputs = self.model(samples)
        orig_target_sizes = torch.stack(
            [torch.tensor(meta['ori_shape'][:2]) for meta in img_metas], dim=0).cuda()
        results = self.postprocessor(outputs, orig_target_sizes)
        assert isinstance(results, list)
        scores = results[0]['scores'].unsqueeze(-1)
        labels = results[0]['labels'].unsqueeze(-1)
        boxes = results[0]['boxes']
        assert scores.size(1) == 1 \
               and labels.size(1) == 1 \
               and boxes.size(1) == 4
        boxes_and_scores = torch.cat([boxes, scores], dim=1)
        results_list = [(boxes_and_scores, labels)]
        bbox_results = [
            bbox2result(det_bboxes,
                        det_labels.squeeze(1),
                        self.model.class_embed.out_features)
            for det_bboxes, det_labels in results_list]
        return bbox_results

    def prepare_samples(self, img, img_metas) -> NestedTensor:
        batch_size = img.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = img.new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
        samples = NestedTensor(img, img_masks)
        return samples

    def prepare_targets(self, img_metas: list, gt_bboxes: list,
                        gt_labels: list, gt_bboxes_ignore=None) -> List[Dict]:
        batch_size = len(img_metas)
        targets = list()
        for i in range(batch_size):
            targets.append(dict(
                boxes=gt_bboxes[i],
                labels=gt_labels[i],
                orig_size=torch.tensor(img_metas[i]['ori_shape'][:2])))
        return targets

    DEFAULT_ARGS = ConfigDict(dict(
        # dataset
        dataset_file='coco',
        # * Backbone
        backbone='resnet50',
        dilation=True,
        position_embedding='sine',
        # * Transformer
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=300,
        pre_norm=False,
        # * Segmentation
        masks=False,
        # Loss
        aux_loss=True,
        # * Matcher
        set_cost_class=2,
        set_cost_bbox=5,
        set_cost_giou=2,
        # * Loss coefficients
        mask_loss_coef=1,
        dice_loss_coef=1,
        cls_loss_coef=2,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        focal_alpha=0.25,
        # others
        device='cuda',
        lr=1e-4,
        lr_backbone=1e-5
    ))


if __name__ == '__main__':
    model = GraftingConditionalDETR()