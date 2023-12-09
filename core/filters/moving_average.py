import numpy as np
from collections import deque

class MovingAverageFilter(object):

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        self._data = deque(maxlen=window_size)
        self._reset()

    def __call__(self, value):
        self._data.append(value)

    def _reset(self):
        self._data.clear()
        self._data.append(0)
    
    @property
    def value(self):
        return np.mean(self._data)