import tensorflow as tf

def fp16_to_fp32_nested(input_nested):
    """Convert fp16 tensors in a nested structure to fp32.

    Args:
        input_nested: A Python dict, values being Tensor or Python list/tuple of
        Tensor or Non-Tensor.

    Returns:
        A Python dict with the same structure as `tensor_dict`,
        with all bfloat16 tensors converted to float32.
    """
    if isinstance(input_nested, tf.Tensor):
        if input_nested.dtype == tf.float16:
            return tf.cast(input_nested, dtype=tf.float32)
        else:
            return input_nested
    elif isinstance(input_nested, (list, tuple)):
        out_tensor_dict = [fp16_to_fp32_nested(t) for t in input_nested]
    elif isinstance(input_nested, dict):
        out_tensor_dict = {
            k: fp16_to_fp32_nested(v) for k, v in input_nested.items()
        }
    else:
        return input_nested
    return out_tensor_dict