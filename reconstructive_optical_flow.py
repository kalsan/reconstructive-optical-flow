import cv2
import numpy as np
from masked_array import MaskedArray
import transform


class ReconstructiveOpticalFlow:
    # initial_img must be single-channel uint8 (grey)
    def __init__(self, initial_img, initial_img_mask):
        init_points = cv2.goodFeaturesToTrack(initial_img,
                                              mask=initial_img_mask,
                                              maxCorners=100,
                                              qualityLevel=0.01,
                                              minDistance=3,
                                              blockSize=7
                                              )
        self.pts_init = MaskedArray(init_points)
        self.pts_old = self.pts_init.copy()
        self.pts_new = None
        self.img_old = initial_img
        self.img_new = None
        self.img_mask_old = initial_img_mask
        self.img_mask_new = None

    def update(self, img_new, img_mask_new):
        self.img_new = img_new
        self.img_mask_new = img_mask_new

        # Apply mask for old points
        self.__set_mask_from_img_mask(self.pts_old, self.img_mask_old)

        # Run optical flow
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.img_old,
            self.img_new,
            self.pts_old.masked_array(nested=True),
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        if new_pts is None:
            return

        # Store new points into a full-sized array but the mask limits the
        # amount of unmasked points such that it is identical to the the
        # amount of old unmasked points
        self.pts_new = self.pts_old.copy()
        self.pts_new.update_unmasked_elements_from_array(new_pts)

        # Now further restrict the new points mask from the new image mask
        self.pts_new.and_mask(
            self.__get_mask_from_img_mask(
                self.pts_new.unmasked_array(), self.img_mask_new
            )
        )

        # Create a mask that contains the mask of pts_old anded with st
        st_enlarged = self.pts_old.mask
        st_enlarged[st_enlarged != 0] = st[:, 0]
        # Apply it onto pts_new as well
        self.pts_new.and_mask(st_enlarged.astype(np.uint8))

        # Apply pts_new's mask to pts_init in order to make them the same size
        self.pts_init.set_mask(self.pts_new.mask)

        # Calculate the transform from orig to new (both constrained)
        T = transform.best_fit_transform(
            self.pts_init.masked_array(),
            self.pts_new.masked_array()
        )

        # Now project the unconstrained init point cloud using this transform.
        # The projected cloud will be the new pts_old with reset mask.
        self.pts_old = MaskedArray(transform.apply_transform(
            self.pts_init.unmasked_array(),
            T
        ))
        self.img_old = self.img_new
        self.img_mask_old = self.img_mask_new

    # This takes a list of coordinates (n x 2) and a binary image mask (h x w)
    # and marks the mask of the point list such that each coordinate is masked
    # iff its point in the binary image mask is masked too
    def __set_mask_from_img_mask(self, pointlist, img_mask):
        pointlist.set_mask(
            self.__get_mask_from_img_mask(pointlist.unmasked_array(), img_mask)
        )

    def __get_mask_from_img_mask(self, coords, img_mask):
        rearranged_coords = np.swapaxes(coords, 0, 1).astype(int)
        rearranged_coords[[0, 1]] = rearranged_coords[[1, 0]]
        for i in range(2):
            rearranged_coords[i][
                rearranged_coords[i] >= img_mask.shape[i]
            ] = img_mask.shape[i] - 1
            rearranged_coords[i][
                rearranged_coords[i] < 0
            ] = 0
        return img_mask[tuple(rearranged_coords)]
