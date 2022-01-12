import tensorflow as tf
import os

from tfcv.config import update_cfg, setup_args, config as cfg
from tfcv.datasets.coco.dataset import Dataset
from tfcv.ops import anchors
from tfcv.models.genelized_rcnn import GenelizedRCNN
from tfcv.ops import roi_ops, spatial_transform_ops, postprocess_ops, training_ops
if __name__ == '__main__':
    config_file = '/home/lin/projects/tfcv/configs/InstanceSeg/mask_rcnn_r50_fpn_1x.yaml'
    config_file = os.path.abspath(config_file)
    params = update_cfg(config_file)
    cfg.from_dict(params)
    model = GenelizedRCNN(cfg)
    model.build([4, 832, 1344, 3], training=True)
    print('model output_specs: \n', model.output_specs)
    dataset = Dataset()
    train_data = dataset.train_fn(batch_size=cfg.train_batch_size)
    it = iter(train_data)
    datapoint = it.get_next()
    print('dataset: ')
    # for k,v in datapoint.items():
    #     print(k, v.shape)
    outputs = model(
        datapoint['images'], 
        datapoint['image_info'], 
        datapoint['gt_classes'], 
        datapoint['gt_boxes'], 
        datapoint['cropped_gt_masks'],
        training=True)
    # print(outputs.shape)    
    def nest_print(inputs, pre='*'):
        for k,v in inputs.items():
            if isinstance(v, dict):
                print(pre, k)
                nest_print(v, pre+'*')
            else:
                try:
                    print(pre, k, v.shape)
                except:
                    print(pre, k, v)
    nest_print(outputs)
    nest_print(model.output_specs)
    # for k, l in model._layers.items():
    #     print(k, l.output_specs)
    # model.save_weights('tmp.npz')
    # print('rpn_box_outputs: ')
    # for k,v in rpn_box_outputs.items():
    #     print(k, v.shape)
    # all_anchors = anchors.Anchors(cfg.min_level, cfg.max_level,
    #                             cfg.anchor.num_scales, cfg.anchor.aspect_ratios,
    #                             cfg.anchor.scale,
    #                             (832, 1344))
    # anchor_boxes = all_anchors.get_unpacked_boxes()
    # training=True
    # if training:
    #     rpn_pre_nms_topn = cfg.rpn.train.pre_nms_topn
    #     rpn_post_nms_topn = cfg.rpn.train.post_nms_topn
    #     rpn_nms_threshold = cfg.rpn.train.nms_threshold

    # else:
    #     rpn_pre_nms_topn = cfg.rpn.test.pre_nms_topn
    #     rpn_post_nms_topn = cfg.rpn.test.post_nms_topn
    #     rpn_nms_threshold = cfg.rpn.test.nms_threshold
    # rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
    #     scores_outputs=rpn_score_outputs,
    #     box_outputs=rpn_box_outputs,
    #     anchor_boxes=anchor_boxes,
    #     image_info=datapoint['image_info'],
    #     rpn_pre_nms_topn=rpn_pre_nms_topn,
    #     rpn_post_nms_topn=rpn_post_nms_topn,
    #     rpn_nms_threshold=rpn_nms_threshold,
    #     rpn_min_size=cfg.rpn.min_size,
    #     bbox_reg_weights=None,
    # )
    # print(rpn_box_scores.shape, rpn_box_rois.shape)


