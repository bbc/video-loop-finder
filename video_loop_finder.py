#! /usr/bin/env/python3

"""Program to find a loop in a repeating video, such as a concentric mosaic dataset

This script will find the best matching frame pair in terms of lowest sum of absolute
pixel differences and localise the end frame relative to the actual beginning/end of the
loop.

For example, if in a concentric mosaic video, the first frame is assumed at 0° and the
closest end frame is found at 359.1°, then the relative position of the latter is
359.1°/360° = 99.75%.
"""

import cv2
import numpy as np
import logging
from enum import Enum
from matplotlib import pyplot as plt


class VideoLoopDirection(Enum):
    CW = 0
    CCW = 1


class VideoLoopFinder:
    """Main class that contains the loop finding logic

    Typical usage:

        vlf = VideoLoopFinder(<path_to_video>, <start_frame_idx>, <duration_hint>)
        end_frame_idx = vlf.find_closest_end_frame()
        relative_end_frame_position = vlf.localise_end_frame()
    """

    def __init__(self, video_path,
                 start_frame_idx=0,
                 duration_hint=None, *,
                 resolution=256,
                 flow_filter_threshold=0.2,
                 debug=False):
        """ Constructor

        Args:
            video_path              – Path to video file or printf-style image sequence
            start_frame_idx         – Index of the frame to match (default: 0)
            duration_hint           – Expected video_duration of video loop in frames
                                      (defaults to video length)
            resolution              – Image width in pixels used in computations. Set to
                                      None to use full original image resolution
                                      (default: 256)
            flow_filter_threshold   – Filter out optical flow vectors that, when chai-
                                      ning forward and backward flows together, do not
                                      map back onto themselves within this number of
                                      pixels. Set to None to disable filtering.
                                      (default: 0.2)
            debug                   — Enable more verbose logging and plot intermediate
                                      results
        """
        self.debug = debug
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                            format="%(levelname)s\t%(message)s")

        # Open video / image sequence and determine its properties
        self.video = cv2.VideoCapture(video_path)
        self.video_duration = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.video_duration == 0:
            self.video_duration = -1
            success = True
            while success:
                self.video_duration += 1
                success, _ = self.video.read()

        if resolution is None:
            resolution = width
        self.resolution = (resolution, int(height / width * resolution))

        if duration_hint is None:
            self.end_frame_hint_idx = self.video_duration - 1
        else:
            self.end_frame_hint_idx = (
                min(self.video_duration, start_frame_idx + duration_hint) - 1
            )

        logging.info(f"Input loaded: video_duration={self.video_duration:.0f}")

        # Seek to start_frame_idx
        self.start_frame_idx = start_frame_idx
        self.start_frame = self._seek(start_frame_idx)

        # Initialise optical flow algorithm
        self.flow_algo = cv2.optflow.createOptFlow_Farneback()
        self.flow_filter_threshold = flow_filter_threshold

        # Determine looping direction
        self.loop_direction = self._find_video_direction()
        logging.info(f"Looping direction appears to be {self.loop_direction.name}")

        # Will be populated by find_closest_end_frame
        self.end_frames = None

    def _seek(self, frame_idx, downsample=True, grayscale=True):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.video.read()
        if not success:
            logging.error(f"Cannot read frame {frame_idx}")

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if downsample:
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)

        return frame

    def _compute_pixel_difference(self, other_frame):
        return np.sum(np.abs(self.start_frame - other_frame))

    def _find_video_direction(self, frame1=None, frame2=None):
        """ Determine the direction the video is spinning in between two frames

        Args:
            frame1      – First frame or its index (defaults to start frame)
            frame2      – Second frame or its index (defaults to start frame + 1)

        Returns:
            VideoDirection.CW or VideoDirection.CCW depending on whether the camera
            motion between frame1 and frame2 has a positive or negative horizontal
            component
        """
        if frame1 is None:
            frame1 = self.start_frame
        elif isinstance(frame1, (np.integer, int)):
            frame1 = self._seek(frame1)

        if frame2 is None:
            frame2 = self._seek(self.start_frame_idx + 1)
        elif isinstance(frame2, (np.integer, int)):
            frame2 = self._seek(frame2)

        flow_forward = self.flow_algo.calc(frame1, frame2, None)

        if self.flow_filter_threshold is not None:
            flow_backward = self.flow_algo.calc(frame2, frame1, None)
            flow_forward = self.filter_optical_flow(
                flow_forward,
                flow_backward,
                self.flow_filter_threshold,
                verbose=self.debug).filled()

        if np.nanmedian(flow_forward[..., 0]) < 0:
            return VideoLoopDirection.CW
        else:
            return VideoLoopDirection.CCW

    def find_closest_end_frame(self, search_range=50):
        """ Find frame most similar to start frame that still lies before it, and sets
        end_grame_idx and end_frames member variables where
            end_frame_idx ← N-1
            end_frames[0] ← frame N-1
            end_frames[1] ← frame N
        Args:
            search_range : int
                Number of frames to check around (start_frame_idx + duration_hint) in
                both directions (default: 50)
        Returns:
            Index of the last frame of the loop (i.e. index N-1)
        """
        idx_from = max(1, self.end_frame_hint_idx - search_range)
        idx_to = min(self.video_duration - 2, self.end_frame_hint_idx + search_range)

        # Iterate over video with 3-frame window, searching for closest match
        prev_frame = None
        curr_frame = None
        next_frame = self._seek(idx_from)
        min_sad = np.inf
        min_idx = idx_from
        min_frames = tuple()  # 3 frames centered on current minimum
        for i in range(idx_from + 1, idx_to + 2):
            # Read new frame
            success, frame = self.video.read()
            frame = cv2.cvtColor(
                       cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA),
                       cv2.COLOR_RGB2GRAY)
            if not success:
                msg = f"Failed to read frame {i}"
                logging.fatal(msg)
                raise RuntimeError(msg)
            # Shift frames along
            prev_frame = curr_frame
            curr_frame = next_frame
            next_frame = frame

            # Test for minimum SAD
            sad = self._compute_pixel_difference(curr_frame)
            if sad and sad < min_sad:
                min_sad = sad
                min_idx = i
                min_frames = prev_frame, curr_frame, next_frame

        if self.loop_direction == self._find_video_direction(min_frames[1],
                                                             self.start_frame):
            self.end_frames = [min_frames[1], min_frames[2]]
            self.end_frame_idx = min_idx
            return min_idx
        else:
            self.end_frames = [min_frames[0], min_frames[1]]
            self.end_frame_idx = min_idx - 1
            return min_idx - 1

    @staticmethod
    def filter_optical_flow(fwd_flow, bwd_flow, threshold, *, verbose=False):
        """Remove unreliable flow vectors from fwd_flow

        Follows the flow from the previous to the next frame (fwd_flow)
        and from the next back to the previous frame (bwd_flow), and
        checks if the final pixel location is within threshold of the
        initial location. If not, the fwd_flow vector at this pixel is
        set to (None,None) to mark it as unreliable.

    Args:
        fwd_flow    – optical flow from previous to next frame
                      which will be filtered
        bwd_flow    – optical flow from next to previous frame
        threshold   – maximum deviation in pixels that the
                      concatenation of fwd_flow aand bwd_flow
                      may exhibit before classified unreliable
    Returns:
        A masked_array the same size as fwd_flow with inconsistent flow values masked
        out
    """
        height, width, depth = fwd_flow.shape

        if bwd_flow.shape != (height, width, depth) or depth != 2:
            raise RuntimeError(
                'Both input flows must have the same size and have 2 channels'
            )

        fwd_flow = np.ma.masked_array(fwd_flow, copy=True, fill_value=np.nan)

        img_coords_x, img_coords_y = np.meshgrid(np.arange(width), np.arange(height))
        img_coords = np.dstack((img_coords_x, img_coords_y)).astype(np.float32)
        coords_in_next = img_coords + fwd_flow
        coords_in_prev = cv2.remap(bwd_flow,
                                   coords_in_next[..., 0],
                                   coords_in_next[..., 1],
                                   cv2.INTER_CUBIC, None) + coords_in_next
        error = np.linalg.norm(coords_in_prev - img_coords, axis=2)
        if verbose:
            plt.hist(error.ravel(), bins=100, range=[0, 2])
            plt.show()

        fwd_flow.mask = error > threshold

        return fwd_flow

    def localise_end_frame(self):
        """Find exact relative location of end frame on the loop

        Returns:
            A float (<= 1.0) that represents the relative location of end frame on the
            loop.
            For example, 1.0 if the end frame perfectly coincides with the start frame,
            or 0.995 if it lies at 99.5%, i.e. 0.5% before the end of the loop.
        """

        if not self.end_frames:
            msg = "find_closest_end_frame must be called before localise_end_frame"
            logging.fatal(msg)
            raise RuntimeError(msg)

        # Compute optical flows 0→(N-1) and 0→N which should point in opposite
        # directions
        flows = [self.flow_algo.calc(self.start_frame,
                                     self.end_frames[0], None),
                 self.flow_algo.calc(self.start_frame,
                                     self.end_frames[1], None)]
        if self.flow_filter_threshold is not None:
            bwd_flows = [self.flow_algo.calc(self.end_frames[0],
                                             self.start_frame, None),
                         self.flow_algo.calc(self.end_frames[1],
                                             self.start_frame, None)]
            flows = [self.filter_optical_flow(flows[i],
                                              bwd_flows[i],
                                              self.flow_filter_threshold)
                     .filled()
                     for i in range(2)]

        # We are only interested in the horizontal components
        xflow_magnitudes = [np.abs(f[..., 0]) for f in flows]
        xflow_magnitude_sum = sum(xflow_magnitudes)

        full_frame_count = self.end_frame_idx - self.start_frame_idx
        fractional_frame_count = np.nanmedian(
                                    xflow_magnitudes[0][xflow_magnitude_sum != 0]
                                    / xflow_magnitude_sum[xflow_magnitude_sum != 0])
        if self.debug:
            plt.imshow(xflow_magnitudes[0] / xflow_magnitude_sum)
            plt.colorbar()
            plt.figure()
            plt.hist((xflow_magnitudes[0] / xflow_magnitude_sum).ravel(), bins=100)
            plt.show()

        return full_frame_count / (full_frame_count + fractional_frame_count)


if __name__ == "__main__":

    vlf = VideoLoopFinder(
        "/home/florians/Videos/VID_2019_09_26_14_02_58_20191015155545.mp4",
        start_frame_idx=33 * 30,
        duration_hint=60 * 30,
        resolution=256,
        flow_filter_threshold=.2
    )
    print(vlf.find_closest_end_frame())
    print(vlf.localise_end_frame())
