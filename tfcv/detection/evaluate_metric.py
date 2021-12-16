import collections
import numpy as np
from pycocotools import cocoeval
import tensorflow as tf

__all__ = ['COCOMetric']

class COCOMetric:
    def __init__(self, 
            include_mask, 
            need_rescale_bboxes, 
            per_category_metrics,
            annotation=None):

        self._include_mask = include_mask
        self._per_category_metrics = per_category_metrics
        self._metric_names = [
            'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1', 'ARmax10',
            'ARmax100', 'ARs', 'ARm', 'ARl'
        ]
        self._required_prediction_fields = [
            'num_detections',
            'detection_classes',
            'detection_scores',
            'detection_boxes'
        ]
        
        self._required_groundtruth_fields = [
            'source_id', 'height', 'width', 'classes', 'boxes', 'num_detections'
        ]
        self._need_rescale_bboxes = need_rescale_bboxes
        if self._need_rescale_bboxes:
            self._required_groundtruth_fields.append('image_info')

        if self._include_mask:
            mask_metric_names = ['mask_' + x for x in self._metric_names]
            self._metric_names.extend(mask_metric_names)
            self._required_prediction_fields.extend(['detection_masks'])
            self._required_groundtruth_fields.extend(['masks'])
        
        self.reset_state()
    
    def reset_state(self):
        """Resets internal states for a fresh run."""
        self._predictions = collections.defaultdict(list)
        self._groundtruths = collections.defaultdict(list)

    def _convert_to_numpy(self, groundtruths, predictions):
        """Converts tesnors to numpy arrays."""
        labels = tf.nest.map_structure(lambda x: x.numpy(), groundtruths)
        numpy_groundtruths = {}
        for key, val in labels.items():
            if isinstance(val, tuple):
                val = np.concatenate(val)
            numpy_groundtruths[key] = val

        outputs = tf.nest.map_structure(lambda x: x.numpy(), predictions)
        numpy_predictions = {}
        for key, val in outputs.items():
            if isinstance(val, tuple):
                val = np.concatenate(val)
            numpy_predictions[key] = val
        if 'source_id' not in numpy_predictions.keys():
            numpy_predictions['source_id'] = numpy_groundtruths['source_id']
        if 'image_info' not in numpy_predictions.keys():
            numpy_predictions['image_info'] = numpy_groundtruths['image_info']


        return numpy_groundtruths, numpy_predictions

    def _process_predictions(self, predictions):
        image_scale = np.tile(predictions['image_info'][:, 2:3, :], (1, 1, 2))
        predictions['detection_boxes'] = (
            predictions['detection_boxes'].astype(np.float32))
        predictions['detection_boxes'] /= image_scale
        if 'detection_outer_boxes' in predictions:
            predictions['detection_outer_boxes'] = (
                predictions['detection_outer_boxes'].astype(np.float32))
            predictions['detection_outer_boxes'] /= image_scale

    def update_state(self, predictions, groundtruths=None):
        groundtruths, predictions = self._convert_to_numpy(groundtruths, predictions)
        for k in self._required_groundtruth_fields:
            if k not in groundtruths:
                raise ValueError(f'Missing the required key `{k}` in groundtruths!')
        
        for k, v in groundtruths.items():
            if k in self._required_groundtruth_fields:
                self._groundtruths[k].append(v)

        if self._need_rescale_bboxes:
            self._process_predictions(predictions)

        for k in self._required_prediction_fields:
            if k not in predictions:
                raise ValueError(f'Missing the required key `{k}` in predictions!')

        for k, v in predictions.items():
            if k in self._required_prediction_fields or k=='source_id':
                self._predictions[k].append(v)

    def result(self):
        """Evaluates detection results, and reset_states."""
        metric_dict = self.evaluate()
        # Cleans up the internal variables in order for a fresh eval next time.
        return metric_dict

    def evaluate(self):
        gt_dataset = convert_groundtruths_to_coco_dataset(
            self._groundtruths)

        coco_gt = COCOWrapper(
            eval_type=('mask' if self._include_mask else 'box'),
            gt_dataset=gt_dataset)
        coco_predictions = convert_predictions_to_coco_annotations(
            self._predictions)
        coco_dt = coco_gt.loadRes(predictions=coco_predictions)
        image_ids = [ann['image_id'] for ann in coco_predictions]

        coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_metrics = coco_eval.stats

        if self._include_mask:
            mcoco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iouType='segm')
            mcoco_eval.params.imgIds = image_ids
            mcoco_eval.evaluate()
            mcoco_eval.accumulate()
            mcoco_eval.summarize()
            mask_coco_metrics = mcoco_eval.stats

        if self._include_mask:
            metrics = np.hstack((coco_metrics, mask_coco_metrics))
        else:
            metrics = coco_metrics
        
        metrics_dict = {}
        for i, name in enumerate(self._metric_names):
            metrics_dict[name] = metrics[i].astype(np.float32)

        return metrics_dict
