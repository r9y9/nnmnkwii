from __future__ import with_statement, print_function, absolute_import

import numpy as np
from scipy import interpolate


def interp1d(f0, kind="slinear"):
    """Coutinuous F0 interpolation from discontinuous F0 trajectory

    This function generates continuous f0 from discontinuous f0 trajectory
    based on :func:`scipy.interpolate.interp1d`. This is meant to be used for
    continuous f0 modeling in statistical speech synthesis
    (e.g., see [1]_, [2]_).

    If ``kind`` = ``'slinear'``, then this does same thing as Merlin does.

    Args:
        f0 (ndarray): F0 or log-f0 trajectory
        kind (str): Kind of interpolation that :func:`scipy.interpolate.interp1d`
            supports. Default is ``'slinear'``, which means linear interpolation.

    Returns:
        1d array (``T``, ) or 2d (``T`` x 1) array: Interpolated continuous f0
        trajectory.

    Examples:
        >>> from nnmnkwii.preprocessing import interp1d
        >>> import numpy as np
        >>> from nnmnkwii.util import example_audio_file
        >>> from scipy.io import wavfile
        >>> import pyworld
        >>> fs, x = wavfile.read(example_audio_file())
        >>> f0, timeaxis = pyworld.dio(x.astype(np.float64), fs, frame_period=5)
        >>> continuous_f0 = interp1d(f0, kind="slinear")
        >>> assert f0.shape == continuous_f0.shape

    .. [1] Yu, Kai, and Steve Young. "Continuous F0 modeling for HMM based
        statistical parametric speech synthesis." IEEE Transactions on Audio,
        Speech, and Language Processing 19.5 (2011): 1071-1079.

    .. [2] Takamichi, Shinnosuke, et al. "The NAIST text-to-speech system for
        the Blizzard Challenge 2015." Proc. Blizzard Challenge workshop. 2015.
    """
    ndim = f0.ndim
    if len(f0) != f0.size:
        raise RuntimeError("1d array is only supported")
    continuous_f0 = f0.flatten()
    nonzero_indices = np.where(continuous_f0 > 0)[0]

    # Nothing to do
    if len(nonzero_indices) <= 0:
        return f0

    # Need this to insert continuous values for the first/end silence segments
    continuous_f0[0] = continuous_f0[nonzero_indices[0]]
    continuous_f0[-1] = continuous_f0[nonzero_indices[-1]]

    # Build interpolation function
    nonzero_indices = np.where(continuous_f0 > 0)[0]
    interp_func = interpolate.interp1d(nonzero_indices,
                                       continuous_f0[continuous_f0 > 0],
                                       kind=kind)

    # Fill silence segments with interpolated values
    zero_indices = np.where(continuous_f0 <= 0)[0]
    continuous_f0[zero_indices] = interp_func(zero_indices)

    if ndim == 2:
        return continuous_f0[:, None]
    return continuous_f0
