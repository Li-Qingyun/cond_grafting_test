_base_ = [
    './conditional_detr/conditional_detr_r50_dc5_8x2_50e_coco.py'
]
model = dict(
    _delete_=True,
    type='GraftingConditionalDETR',
    # dataset
    dataset_file='coco',
    # * Backbone
    backbone='resnet50',
    dilation=True,
    position_embedding='sine',
    # * Transformer
    enc_layers=6,  # TODO
    dec_layers=6,  # TODO
    dim_feedforward=2048,  # TODO
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    num_queries=300,  # TODO
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
    lr_backbone=1e-5)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(continuous_categories=True),
    val=dict(continuous_categories=True),
    test=dict(continuous_categories=True)
)
custom_imports = dict(
    imports='custom',
    allow_failed_imports=False)

# yapf:enable
custom_hooks = []  # to delete NumClassCheckHook