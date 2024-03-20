import gunpowder as gp
import random
import numpy as np
from scipy.ndimage import gaussian_filter


class SmoothArray(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def _process_section(self, array_sec):
        sigma = random.uniform(0.0, 1.0)
        return np.array(gaussian_filter(array_sec, sigma=sigma)).astype(array_sec.dtype)

    def process(self, batch, request):
        array = batch[self.array].data

        assert len(array.shape) == 3 or len(array.shape) == 2

        if len(array.shape) == 2:
            array = self._process_section(array)

        if len(array.shape) == 3:
            for z in range(array.shape[0]):
                array[z] = self._process_section(array[z])

        batch[self.array].data = array
