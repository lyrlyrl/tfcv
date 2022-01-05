import pprint

__all__ = ['NanTrainLoss']

class NanTrainLoss(Exception):
    def __init__(self, period, metrics) -> None:
        message = pprint.pformat(metrics)
        prefix = f'detect nan loss at {period} with metrics below:\n'
        super().__init__(message=prefix+message)

class LayerNotBuilt(Exception):
    pass