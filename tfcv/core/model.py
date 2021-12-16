import os
import logging
import numpy as np
import tensorflow as tf

from tfcv.utils.http import get_file

ALL_PRETRAINED_WEIGHTS = {
    'resnet50_imagenet':{
        'sha256': '42352458018f796aa104c4729403c85d55ad257c56ee60cf138369cdc7b01d41',
        'urls': [
            'http://192.168.1.20:8000/resnet50_imagenet.npz',
            'http://192.168.50.12:8080/pretrained_weights/resnet50_imagenet.npz',
            'http://www.linstudio.work:6680/pretrained_weights/resnet50_imagenet.npz'
        ]
    }
}

def _unprefix(name, prefix):
    names = name.split('/')
    if names[0]==prefix:
        return '/'.join(names[1:])
    else:
        return name


class Model(tf.keras.Model):
    
    def load(self, key_word):
        logger = logging.getLogger(__name__)
        assert key_word in ALL_PRETRAINED_WEIGHTS

        fp = None
        for url in ALL_PRETRAINED_WEIGHTS[key_word]['urls']:
            try:
                fp = get_file(
                    key_word+'.npz',
                    url,
                    file_hash=ALL_PRETRAINED_WEIGHTS[key_word]['sha256'],
                    cache_subdir='pretrained_weights',
                    hash_algorithm='sha256'
                )
                break
            except:
                continue
        if fp is None:
            raise
        else:            
            logger.debug(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            logger.debug(f'@@@@@@@@start loading weights from {key_word} to {self.name}@@@@@@@@')
            logger.debug(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            r = np.load(fp)
            for v in self.variables:
                name = v.name
                if name in r:
                    v.assign(r[name])
                    logger.debug(f'loading {name} successed with shape of {r[name].shape}')
                elif name.split(':')[0] in r:
                    name = name.split(':')[0]
                    v.assign(r[name])
                    logger.debug(f'loading {name} successed with shape of {r[name].shape}')
                elif _unprefix(name.split(':')[0], key_word) in r:
                    name = _unprefix(name.split(':')[0], key_word)
                    v.assign(r[name])
                    logger.debug(f'loading {name} successed with shape of {r[name].shape}')
                elif _unprefix(name.split(':')[0], key_word.split('_')[0]) in r:
                    name = _unprefix(name.split(':')[0], key_word.split('_')[0])
                    v.assign(r[name])
                    logger.debug(f'loading {name} successed with shape of {r[name].shape}')
                else:
                    logging.warning(f'loading {name} failed')
        logger.debug(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        logger.debug(f'@@@@@@@@finished loading weights from {key_word} to {self.name}@@@@@@@@')
        logger.debug(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')