import math
import tensorflow as tf
import os

import tfcv
from tfcv.utils.progress import get_tqdm

from tfcv.evaluate.metric import COCOEvaluationMetric, process_predictions
from tfcv.ops import preprocess_ops
from tfcv.ops import anchors
from tfcv.datasets.coco.dataset_parser import preprocess_image

class DetectionRunner(tfcv.Runner):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        eval_file = os.path.join(self._params.data.dir, self._params.data.val_json)
        eval_file = os.path.expanduser(eval_file)
        self.coco_metric = COCOEvaluationMetric(eval_file, self._params.include_mask)
    def evaluate(self, dataset):
        results = []
        for dp in get_tqdm(dataset):
            results.append(self._validation_op(dp))
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

class DetectionExporter(tfcv.Exporter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def preprocess(self, image, resized=False):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = preprocess_ops.normalize_image(
            image, 
            self._params.data.pixel_std,
            self._params.data.pixel_mean)
        if not resized:
            image, image_info, _, _ = preprocess_image(
                image,
                None,
                None,
                image_size=self._params.data.image_size,
                max_level=self._params.max_level
            )
        return image, image_info
    
    def inference_step(self, image):
        image, image_info = self.preprocess(image)
        detections = self._model.call(
            image, image_info, training=False
        )
        detections['image_info'] = image_info
        return detections
    
    
