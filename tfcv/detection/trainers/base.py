import tensorflow as tf
import os

import tfcv

from tfcv.detection.evaluate.metric import COCOEvaluationMetric, process_predictions

class DetectionTrainer(tfcv.Trainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        eval_file = os.path.join(self._params.data.dir, self._params.data.val_json)
        eval_file = os.path.expanduser(eval_file)
        self.coco_metric = COCOEvaluationMetric(eval_file, self._params.include_mask)
    def evaluate(self, dataset):
        results = []
        for dp in dataset:
            results.append(self._inference_op(dp))
        def _merge(*args):
            return tf.concat(args, 0).numpy()
        results = tf.nest.map_structure(_merge, *results)
        for k, v in results.items():
            print(k, v.shape)
        predictions = process_predictions(results)
        metric = self.coco_metric.predict_metric_fn(predictions)
        for k, v in metric.items():
            metric[k] = v.tolist()
        return metric