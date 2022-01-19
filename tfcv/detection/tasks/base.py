from tfcv.core import Task

from tfcv.detection.models.genelized_rcnn import GenelizedRCNN

class DetectionTask(Task):
    def __init__(self, params):
        self._params = params
    def create_model(self):
        if self._params.meta_arch == 'genelized_rcnn':
            return GenelizedRCNN(self._params)
        else:
            raise