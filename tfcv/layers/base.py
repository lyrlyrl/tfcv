import abc
import logging

import tensorflow as tf

class Layer(tf.Module, metaclass = abc.ABCMeta):

    default_name = 'defualt_layer'

    def __init__(self, name=None):
        if not name:
            name = self.default_name
        super(Layer, self).__init__(name=name)
        self._output_specs = None
        self._layers = dict()
        self._built = False

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and k != 'name' and not k.startswith('_'):
                    setattr(self, k, v)

    @property
    def output_specs(self):
        if not self._built:
            logging.warn('layer not built yet')
        return self._output_specs

    @property
    def built(self):
        return self._built

    def build(self, *args, **kwargs):
        self._build(*args, **kwargs)
        self._built = True
        assert self._output_specs is not None
    
    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        pass
    
    def inference_post_process(self, inputs):
        return inputs
    
    def train_forward(self, *args, **kwargs):
        kwargs['training'] = True
        return self.call(*args, **kwargs)

    def inference_forward(self, *args, **kwargs):
        
        kwargs['training'] = False
        print(args, kwargs)
        model_outputs = self.call(*args, **kwargs)
        return self.inference_post_process(model_outputs)

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        return 1
    
    def get_layer(self, name: str):
        if name in self._layers.keys():
            return self._layers[name]
        elif name.split('/')[0] in self._layers.keys() and len(name.split('/')) > 1:
            return self._layers[name.split('/')[0]].get_layer('/'.join(name.split('/')[1:]))
        else:
            raise KeyError(f'layer name {name} not found in {self.name}')
