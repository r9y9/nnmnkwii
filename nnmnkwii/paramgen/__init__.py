from __future__ import with_statement, print_function, absolute_import


try:
    import bandmat
except ImportError:
    raise ImportError("""
    `bandmat` is required for the `paramgen` module but not found.
    Please install it manually by:
    `pip install bandmat`.
    If you see installation errors, please report to https://github.com/MattShannon/bandmat.
    """)

from ._mlpg import build_win_mats, mlpg, mlpg_grad, unit_variance_mlpg_matrix
from ._mlpg import reshape_means, full_window_mat


__all__ = ['build_win_mats', 'mlpg', 'mlpg_grad', 'unit_variance_mlpg_matrix',
           'reshape_means', 'full_window_mat']
