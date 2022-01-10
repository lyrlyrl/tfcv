import abc
import logging
import os
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from tfcv.layers.utils import need_build
NOT_AUTO_INIT = ['self', 'name', 'trainable']

def _unprefix(name, prefix):
    names = name.split('/')
    if names[0]==prefix:
        return '/'.join(names[1:])
    else:
        return name
class Layer(tf.Module, metaclass = abc.ABCMeta):

    default_name = 'defualt_layer'

    def __init__(self, trainable=True, name=None):
        if not name:
            name = self.default_name
        super(Layer, self).__init__(name=name)
        self._output_specs = None
        self._layers = OrderedDict()
        self._metrics = dict()
        self._built = False
        self._trainable = trainable

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k not in NOT_AUTO_INIT and not k.startswith('_'):
                    setattr(self, k, v)

    @property
    def output_specs(self):
        if not self._built:
            logging.warn('layer not built yet')
        return self._output_specs

    @property
    def built(self):
        return self._built

    @property
    def trainable(self):
        return self._trainable
    
    @trainable.setter
    def trainable(self, value):
        assert isinstance(value, bool)
        self.trainable = value
    
    @abc.abstractmethod
    def compute_output_specs(self, input_shape):
        pass
    
    def build(self, *args, **kwargs):
        self._build(*args, **kwargs)
        self._built = True
        # assert self._output_specs is not None
    
    def _build(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwds):
        return self.call(*args, **kwds)

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
    
    @property
    def layers(self):
        return self._layers
    
    @property
    def nest_layers(self):
        return tf.nest.flatten(self._layers)

    @property
    def metrics(self):
        return self.metrics
    
    @property
    def all_metrics(self):
        metrics = tf.nest.flatten(self._metrics)
        for l in self.nest_layers:
            metrics.extend(l.all_metrics)
        return metrics

    @need_build
    def load_weights(self, file_path: str, prefix=None, skip_mismatch=False):
        assert file_path.endswith('.npz')
        if prefix == None:
            prefix = self.name
        r = np.load(file_path)
        debug_loading = os.getenv('DEBUG_LOADING_WEIGHTS', '0') == '1'
        logger = logging.getLogger('model_npz_loader')
        if debug_loading:
            logger.info(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            logger.info(f'@@@@@@@@start loading weights from {file_path} to {self.name}@@@@@@@@')
            logger.info(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        
        for v in self.variables:
            name = v.name
            if name in r:
                v.assign(r[name])
                if debug_loading:
                    logger.info(f'loading {name} successed with shape of {r[name].shape}')
            elif name.split(':')[0] in r:
                name = name.split(':')[0]
                v.assign(r[name])
                if debug_loading:
                    logger.info(f'loading {name} successed with shape of {r[name].shape}')
            elif _unprefix(name.split(':')[0], prefix) in r:
                name = _unprefix(name.split(':')[0], prefix)
                v.assign(r[name])
                if debug_loading:
                    logger.info(f'loading {name} successed with shape of {r[name].shape}')
            elif _unprefix(name.split(':')[0], prefix.split('_')[0]) in r:
                name = _unprefix(name.split(':')[0], prefix.split('_')[0])
                v.assign(r[name])
                if debug_loading:
                    logger.info(f'loading {name} successed with shape of {r[name].shape}')
            else:
                if debug_loading:
                    logger.warning(f'loading {name} failed')
        if debug_loading:
            logger.info(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            logger.info(f'@@@@@@@@finished loading weights from {prefix} to {self.name}@@@@@@@@')
            logger.info(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    @need_build
    def save_weights(self, file_path, exclude_pattern=None):
        assert file_path.endswith('.npz')
        values = {
            v.name: v.numpy()
            for v in self.variables
        }
        np.savez(file_path, **values)