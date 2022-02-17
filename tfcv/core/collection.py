__all__ = ['G']

class GlobalHolder:
    trainable_variables = []
    train_metrics = dict()
    evaluate_metrics = dict()
    is_training = True
    def __init__(self):
        self._collections = dict()
    def add_list_collection(self, name):
        assert name not in self._collections
        self._collections[name] = []
    def add_dict_collection(self, name):
        assert name not in self._collections
        self._collections[name] = dict()
    def add_bool_flag(self, name, default=True):
        self._collections[name] = default
    def __getattr__(self, name):
        assert name in self._collections
        return self._collections[name]

G = GlobalHolder()
