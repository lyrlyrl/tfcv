from typing import List

def need_build(f):
    def wrapper(self, *args, **kwargs):
        if self.built:
            return f(self, *args, **kwargs)
        else:
            raise ValueError(f'function {self.name}.{f.__name__} need build ahead')
    return wrapper

def compute_sequence_output_specs(layers: List, inputs):
    for l in layers:
        inputs = l.compute_output_specs(inputs)
    return inputs