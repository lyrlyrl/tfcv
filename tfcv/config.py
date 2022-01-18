import pprint
import collections
import os
import yaml

class AttrDict():

    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        if name.startswith('_'):
            # Do not mess with internals. Otherwise copy/pickle will fail
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1, width=100, compact=True)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def from_dict(self, d):
        self.freeze(False)
        for k, v in d.items():
            self_v = getattr(self, k)

            if isinstance(v, dict):
                self_v.from_dict(v)
            else:
                setattr(self, k, v)

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for k in keylist[:-1]:
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self, freezed=True):
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()

config = AttrDict()

_C = config

def _dict_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.abc.Mapping):
            tmp = _dict_update(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

def update_cfg(config_file):
    assert os.path.isfile(config_file)
    with open(config_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.CLoader)
    if '_base' in params:
        new_config_file = os.path.normpath(os.path.join(config_file, '..', params['_base']))
        params = _dict_update(update_cfg(new_config_file), params)
        del params['_base']
    return params

def setup_args(arguments, cfg):
    if arguments.seed:
        cfg.seed = arguments.seed
    cfg.xla = arguments.xla
    cfg.amp = arguments.amp
    cfg.solver.steps_per_loop = arguments.steps_per_loop
    cfg.solver.evaluate_interval = arguments.evaluate_interval
    cfg.num_gpus = len(arguments.gpu_ids)
    if arguments.config_override:
        cfg.update_args(arguments.config_override)