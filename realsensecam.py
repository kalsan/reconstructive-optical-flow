import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

__realsensecam_instance = None


def realsensecam(*init_params):
    global __realsensecam_instance
    if __realsensecam_instance is None:
        print("Starting camera")
        try:
            __realsensecam_instance = RealsenseCam(*init_params)
        except:
            print("Could not initialize RealSense camera! Make sure it is supported by pyrealsense2. Try re-plugging it (maybe to a different USB port).")
            raise
    return __realsensecam_instance


class RealsenseCam:
    W = 640
    H = 480

    CROP_TOP_LEFT = (100, 100)
    CROP_BOTTOM_RIGHT = (505, 395)

    def __init__(self):
        # Initialize RealSense intrinsics
        self.diagonal = np.linalg.norm((self.W, self.H))
        self.pipeline = rs.pipeline()
        self.aligner = rs.align(rs.stream.color)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, 30)
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()

        # Start up camera
        profile = self.pipeline.start(self.config)

        # Set camera options
        sensor = profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 1)
        # sensor.set_option(rs.option.exposure, 5000)

        # Acquire an initial set of frames used for calibration
        # Flush 10 frames to get the Intel temporal filter warmed up
        for i in range(30):
            self.__acquire_raw_aligned()

        # Save a snapshot of the background for later subtraction, blur it for denoising purposes
        self.__depth_background = ndimage.gaussian_filter(self.__depth_raw_aligned, 20)

    def __acquire_raw_aligned(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.aligner.process(frames)

        # Store color bitmap from the regular sensor as a numpy array (suitable for OpenCV)
        self.bgr = np.asanyarray(aligned_frames.get_color_frame().get_data())

        # Get denoised distance bitmap of depth (larger (brighter) pixel is further from sensor)
        # In this intermediate result, objects have different coordinates than in the bgr image
        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filling_filter.process(depth_frame)
        self.__depth_raw_aligned = np.asanyarray(depth_frame.get_data())

    def acquire_frames(self):
        # First, acquire fresh frames and retrieve the aligned but still raw depth
        self.__acquire_raw_aligned()
        depth = self.__depth_raw_aligned

        # Remove the background captured in the first picture
        depth = self.__depth_background - depth

        # Finally store a uint8 bitmap where 0 is the table height
        depth[depth < 0] = 0
        depth[depth > 255] = 0  # Values over 255 are either overflow results or not interesting to us, classify as background
        self.depth_processed = np.array(depth, np.uint8)
        self.depth_blurred = cv2.GaussianBlur(self.depth_processed, (19, 19), 0)

    # Show a curve visualizing the distribution of the depth across all pixels
    def visualize_depth_distribution(self):
        x, y = np.unique(self.depth_processed, return_counts=True)
        plt.plot(x, y)
        plt.show()

    # You MUST call this upon program exit (even on exception), otherwise the cam will fail to start the next time.
    def stop(self):
        self.pipeline.stop()
        print("Cam stopped")
