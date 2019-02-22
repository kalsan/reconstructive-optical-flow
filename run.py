import cv2
import numpy as np
import icp
from realsensecam import realsensecam


def acquire_masks():
    # Find the hand
    _, depth_th_hand = cv2.threshold(
        realsensecam().depth_blurred, 2, 255, cv2.THRESH_BINARY
    )

    # Create the hand mask, the largest depth contour is assumed to be the hand
    hand_cnt = None
    hand_mask = np.zeros((realsensecam().H, realsensecam().W), np.uint8)
    _, contours, _ = cv2.findContours(
        depth_th_hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        largest_cnt = max(contours, key=lambda c: cv2.contourArea(c))
        if cv2.contourArea(largest_cnt) > 500:
            hand_cnt = largest_cnt
    if hand_cnt is not None:
        cv2.drawContours(hand_mask, [hand_cnt], 0, 255, cv2.FILLED)

    # Create the shape mask
    # Get mask by filtering by color saturation in HSV color space
    hsv = cv2.cvtColor(realsensecam().bgr, cv2.COLOR_BGR2HSV)
    shape_mask_unfiltered = cv2.inRange(
        hsv, np.array([0, 115, 0]), np.array([255, 255, 255])
    )
    _, shape_contours, _ = cv2.findContours(
        cv2.bitwise_and(shape_mask_unfiltered,
                        shape_mask_unfiltered, mask=(255 - hand_mask)),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    largest_shape_cnt = max(shape_contours, key=lambda c: cv2.contourArea(c))

    # Create a mask that should only contain the largest shape without hand
    shape_mask_filtered = np.zeros_like(hand_mask)
    cv2.drawContours(shape_mask_filtered, [
                     largest_shape_cnt], 0, 255, cv2.FILLED)
    shape_mask_filtered[shape_mask_filtered != 0] = 1
    return shape_mask_filtered, largest_shape_cnt

realsensecam()
while True:
    # Read a new frame from the camera
    realsensecam().acquire_frames()
    shape_mask_filtered, largest_shape_cnt = acquire_masks()

    pretty_image = realsensecam().bgr.copy()
    colored_overlay = np.zeros_like(pretty_image)
    colored_overlay[shape_mask_filtered == 0] = (0, 0, 255)
    pretty_image = cv2.addWeighted(colored_overlay, 0.5, pretty_image, 1, 0)
    cv2.drawContours(pretty_image, [largest_shape_cnt], 0, (255, 255, 255))
    cv2.imshow("Overview", pretty_image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

realsensecam().stop()
