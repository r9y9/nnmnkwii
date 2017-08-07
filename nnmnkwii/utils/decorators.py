from __future__ import division, print_function, absolute_import

import numpy as np
try:
    from inspect import getfullargspec
except:
    # python 2.7
    from inspect import getargspec as getfullargspec
from decorator import decorator


@decorator
def apply_along_last_axis_2d(func, *args, **kwargs):
    """Apply function along last axis for matrix

    This is used for extending matrix-to-matrix operations to 3darray-to-2darray
    operations. This basically does the following thing in a convenient way:

    ```py
    np.apply_along_axis(func, input_matrix, -1, *args, **kwargs)
    ```

    Note: The decorator assumes that the first argment of the function is the
    input matrix (2d numpy array).
    """

    # Get first arg
    first_arg_name = getfullargspec(func)[0][0]
    has_positional_arg = len(args) > 0
    input_arg = args[0] if has_positional_arg else kwargs[first_arg_name]

    if input_arg.ndim == 2:
        ret = func(*args, **kwargs)
    else:
        # TODO?
        ret = np.apply_along_axis(func, -1, *args, **kwargs)

    return ret
