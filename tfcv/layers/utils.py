from typing import List

def need_build(f):
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.built:
            return f(*args, **kwargs)
        else:
            raise ValueError(f'function {self.name}.{f.__name__} need build ahead')
    return wrapper

def compute_sequence_output_specs(layers: List, inputs):
    for l in layers:
        inputs = l.compute_output_specs(inputs)
    return inputs

def build_layers(layers: List, input_shape):
    for l in layers:
        l.build(input_shape)
        input_shape = l.output_specs