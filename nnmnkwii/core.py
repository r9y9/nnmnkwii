from __future__ import with_statement, print_function, absolute_import


import sklearn
import numpy as np


# All transfomers (feature extractions, aliging features, etc)
TransformerMixin = sklearn.base.TransformerMixin


class VariableLengthArrayTransfomer(TransformerMixin):
    """Transforms each ulterance

    Multiple ulterances are represented as a list of variable length arrays
    """

    def transform(self, X):
        # ulterance-wise transform
        if X.dtype == np.object:
            return np.array(list(map(self.do_transform, X)))
        else:
            return self.do_transform(X)

    def inverse_transform(self, X):
        if X.dtype == np.object:
            return np.array(list(map(self.do_inverse_transform, X)))
        else:
            return self.do_inverse_transform(X)

    def do_transform(self):
        raise NotImplemented("")

    def do_inverse_transform(self):
        raise NotImplemented("")


Aligner = TransformerMixin
