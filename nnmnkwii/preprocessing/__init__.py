# coding: utf-8
from __future__ import division, print_function, absolute_import

from .generic import mulaw, inv_mulaw, mulaw_quantize, inv_mulaw_quantize
from .generic import preemphasis, inv_preemphasis
from .generic import delta_features, trim_zeros_frames, remove_zeros_frames
from .generic import adjust_frame_length, adjust_frame_lengths
from .generic import scale, inv_scale
from .generic import minmax_scale_params, minmax_scale, inv_minmax_scale
from .generic import meanvar, meanstd, minmax
from .f0 import interp1d
from .modspec import modspec, modphase, inv_modspec, modspec_smoothing

# to be removed at v0.1.0
adjast_frame_length = adjust_frame_length
adjast_frame_lengths = adjust_frame_lengths

__all__ = ['mulaw', 'inv_mulaw', 'mulaw_quantize', 'inv_mulaw_quantize',
           'preemphasis', 'inv_preemphasis', 'delta_features',
           'trim_zeros_frames', 'remove_zeros_frames',
           'adjust_frame_length', 'adjust_frame_lengths',
           'adjast_frame_length', 'adjast_frame_lengths',  # to be removed
           'scale', 'inv_scale', 'minmax_scale_params', 'minmax_scale',
           'inv_minmax_scale', 'meanvar',
           'meanstd', 'minmax', 'interp1d',
           'modspec', 'modphase', 'inv_modspec', 'modspec_smoothing']
