from functools import reduce
from operator import mul


def flatten(shape, keep):
    fixed_shape = shape
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] for i in range(start, len(fixed_shape))]
    return out_shape

def reconstruct(tensor_shape, ref_shape, keep):
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    return target_shape


tensor_shape = flatten([60, 1, 161, 20, 200], 1)
tensor_shape_linear = [193200, 1]
ref_shape = [60, 1, 161, 20, 200]

new_tensor_shape = reconstruct(tensor_shape_linear, ref_shape, 1)
print(new_tensor_shape)
