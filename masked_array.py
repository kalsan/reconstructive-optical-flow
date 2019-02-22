import numpy as np


# Initialize either with an array of arrays
# e.g. [[1,2], [3,4], ...]
# or with another layer of nesting like in opencv contours
# e.g [[[1,2]], [[3,4]], ...]
class MaskedArray:
    def __init__(self, array, mask=None):
        array = np.array(array)
        assert 2 <= len(array.shape) <= 3
        if len(array.shape) == 3:
            assert array.shape[1] == 1
        if mask is not None:
            assert len(mask.shape) == 1
            assert len(array) == len(mask)

        # If the array is nested, save as 2d array
        if len(array.shape) == 3:
            array = array[:, 0]
        self.array = np.array(array)

        # Create mask if needed
        if mask is None:
            mask = [1] * len(array)
        self.mask = np.array(mask)

    def masked_array(self, nested=False):
        return self.__nest(self.array[self.mask != 0].copy(), nested)

    def unmasked_array(self, nested=False):
        return self.__nest(self.array.copy(), nested)

    def set_mask(self, new_mask):
        assert len(new_mask) == len(self.mask)
        assert len(new_mask.shape) == 1
        self.mask = np.array(new_mask)

    def and_mask(self, mask_to_and):
        assert len(mask_to_and) == len(self.mask)
        assert len(mask_to_and.shape) == 1
        self.mask *= np.array(mask_to_and)

    def copy(self):
        return MaskedArray(self.array, self.mask)

    def update_unmasked_elements_from_array(self, array):
        assert 2 <= len(array.shape) <= 3
        if len(array.shape) == 3:
            assert array.shape[1] == 1
            array = array[:, 0]
        self.array[self.mask != 0] = np.array(array)

    def __nest(self, arr, nested):
        if nested:
            return arr[:, None]
        return arr
