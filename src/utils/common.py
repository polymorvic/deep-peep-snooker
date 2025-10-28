from abc import ABC, abstractmethod
from collections.abc import Hashable as SupportsHash

import numpy as np

type array_like = np.ndarray | NumpyImage

class NumpyImage(np.ndarray):
    """
    A lightweight wrapper around `numpy.ndarray` for easier image shape handling.

    Provides convenient properties to access image dimensions:
    - `height`: number of rows
    - `width`: number of columns
    - `depth`: number of channels (defaults to 1 if not present)

    Fully compatible with OpenCV and other libraries that expect a standard
    NumPy array, since it is implemented as a view of the original array.
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    @property
    def width(self):
        return self.shape[1] if len(self.shape) > 1 else 1
    
    @property
    def height(self):
        return self.shape[0]
    
    @property
    def depth(self):
        return self.shape[2] if len(self.shape) > 2 else 1
    
    def as_array(self):
        """Convert back to regular numpy array for compatibility"""
        return np.asarray(self)


class Hashable(ABC):
    @abstractmethod
    def _key_(self) -> SupportsHash:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self._key_())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._key_() == other._key_()
