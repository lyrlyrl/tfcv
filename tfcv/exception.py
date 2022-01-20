import pprint

__all__ = ['NanTrainLoss', 'ManuallyInterrupt']

class NanTrainLoss(Exception):
    def __init__(self, period, metrics) -> None:
        super().__init__(self)
        prefix = f'detect nan loss at {period} with metrics below:\n'
        self.message = prefix+pprint.pformat(metrics)
    def __str__(self) -> str:
        return self.message

class ManuallyInterrupt(Exception):
    def __init__(self, source):
        super().__init__(self)
        self.source = source
    def __str__(self) -> str:
        return self.message