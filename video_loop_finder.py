#! /usr/bin/env/python3

import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")


class VideoLoopFinder:
    """Find a loop in a repeating video

    This class looks for the frame in a video that best matches the first frame
    in terms of lowest SAD.
    """

    def __init__(self, video_path, *, start_frame_idx=0, duration_hint=None):
        """ Constructor

        Args:
            video_path      – Path to video file or printf-style image sequence
            start_frame_idx – Index of the frame to match (default: 0)
            duration_hint   – Expected video_duration of video loop in frames
                              (defaults to video length)
        """
        # Open video / image sequence and determine its length
        self.video = cv2.VideoCapture(video_path)
        self.video_duration = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.video_duration == 0:
            self.video_duration = -1
            success = True
            while success:
                self.video_duration += 1
                success, _ = self.video.read()

        if duration_hint is None:
            self.end_frame_idx = self.video_duration - 1
        else:
            self.end_frame_idx = (
                min(self.video_duration, start_frame_idx + duration_hint) - 1
            )

        logging.info(f"Input loaded: video_duration={self.video_duration:.0f}")

        # Seek to start_frame_idx
        self.start_frame_idx = start_frame_idx
        self.start_frame = self._seek(start_frame_idx)

    def _seek(self, frame_idx):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.video.read()
        if not success:
            logging.error(f"Cannot read frame {frame_idx}")
        else:
            return frame

    def _compute_pixel_difference(self, other_frame):
        return np.sum(np.abs(self.start_frame - other_frame))

    def find_closest_end_frame(self, search_range=50):
        """Find range that represents the loop.

        Args:
            search_range    — Range of frames to check around
                              (start_frame_idx + duration_hint) (default: 50)
        Returns:
            Tuple of start (inclusive) and end (exclusive) indices that represents the
            full loop
        """
        _from = max(0, self.end_frame_idx - search_range)
        _to = min(self.video_duration, self.end_frame_idx + search_range)
        min_idx = _from
        frame = self._seek(_from)
        min_sad = self._compute_pixel_difference(frame)
        for i in range(_from + 1, _to):
            success, frame = self.video.read()
            if not success:
                logging.error(f"Failed to read frame {i}")
                continue
            sad = self._compute_pixel_difference(frame)
            if sad < min_sad:
                min_sad = sad
                min_idx = i

        return self.start_frame_idx, min_idx

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
    """
        height, width, depth = fwd_flow.shape

        if bwd_flow.shape != (height, width, depth) or depth != 2:
            raise RuntimeError(
                'Both input flows must have the same size and have 2 channels'
            )

        fwd_flow = np.copy(fwd_flow)

        img_coords_x, img_coords_y = np.meshgrid(np.arange(width), np.arange(height))
        img_coords = np.dstack((img_coords_x, img_coords_y)).astype(np.float32)
        coords_in_next = img_coords + fwd_flow
        coords_in_prev = cv2.remap(bwd_flow,
                                   coords_in_next[..., 0],
                                   coords_in_next[..., 1],
                                   cv2.INTER_CUBIC, None) + coords_in_next
        error = np.linalg.norm(coords_in_prev - img_coords, axis=2)
        if verbose:
            from matplotlib import pyplot as plt
            plt.hist(error.ravel(), bins=100, range=[0, 2])
            plt.show()

        fwd_flow[error > threshold] = np.nan

        return fwd_flow


if __name__ == "__main__":
    vlf = VideoLoopFinder(
        "/hd_data/Dropbox (BBC)/Videos/VID_2019_09_26_14_02_58_20191015155545.mp4",
        start_frame_idx=33 * 30,
        duration_hint=60 * 30,
    )
    print(vlf.find_closest_end_frame())
