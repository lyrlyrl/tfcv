import logging
import numpy as np
import tensorflow as tf
import os

from tfcv.utils.http import get_file

ALL_PRETRAINED_WEIGHTS = {
    'resnet50_imagenet':{
        'sha256': '42352458018f796aa104c4729403c85d55ad257c56ee60cf138369cdc7b01d41',
        'urls': [
            'http://192.168.50.12:8080/pretrained_weights/resnet50_imagenet.npz',
            'http://139.155.0.247:6680/pretrained_weights/resnet50_imagenet.npz'#TODO use domain name
        ]
    }
}

def _unprefix(name, prefix):
    names = name.split('/')
    if names[0]==prefix:
        return '/'.join(names[1:])
    else:
        return name

def _from_npz(
    model: tf.keras.Model, 
    npz_file: str,
    prefix: str = None) -> tf.keras.Model:

    r = np.load(npz_file)
    debug_loading = os.getenv('DEBUG_LOADING_WEIGHTS', '0') == '1'
    logger = logging.getLogger('model_npz_loader')
    if debug_loading:
        logger.info(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        logger.info(f'@@@@@@@@start loading weights from {prefix} to {model.name}@@@@@@@@')
        logger.info(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    
    for v in model.variables:
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
        logger.info(f'@@@@@@@@finished loading weights from {prefix} to {model.name}@@@@@@@@')
        logger.info(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

def load_npz(model, weights):
    
    if weights.endswith('.npz'):
        _from_npz(model, weights)

    else:
        assert weights in ALL_PRETRAINED_WEIGHTS

        fp = None
        for url in ALL_PRETRAINED_WEIGHTS[weights]['urls']:
            try:
                fp = get_file(
                    weights+'.npz',
                    url,
                    file_hash=ALL_PRETRAINED_WEIGHTS[weights]['sha256'],
                    cache_subdir='pretrained_weights',
                    hash_algorithm='sha256'
                )
                break
            except:
                continue
        if fp is None:
            raise
        else:            
            _from_npz(model, fp)