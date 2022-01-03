from tfcv.layers.base import Layer

def need_build(f):
    def wrapper(self: Layer, *args, **kwargs):
        if self.built:
            return f(self, *args, **kwargs)
        else:
            raise ValueError(f'function {self.name}.{f.__name__} need build ahead')
    return wrapper