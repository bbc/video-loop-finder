#! /usr/bin/env python

"""Video Loop Finder

USAGE:
    video_loop_finder.py [options] VIDEO_PATH [START_FRAME_IDX [DURATION_HINT]]

ARGUMENTS:
    VIDEO_PATH          Path to a video file or printf-style escaped path to image
                        sequence, e.g. '/path/to/image%04d.png'
    START_FRAME_IDX     Index of first frame of loop [default: 0]
    DURATION_HINT       Estimated duration of loop in frames [default: video duration]

OPTIONS:
    -r RANGE --range=RANGE              Search for end frame ±RANGE frames around
                                        START_FRAME + DURATION_HINT [default: 50]
    -b RANGE --match-brightness=RANGE   Adjust START_FRAME (and matching end frame)
                                        position within ±RANGE such that the average
                                        brightness difference between them is mimimum
                                        [default: 0]
    -w WIDTH --width=WIDTH              Image width in pixels used in computations. Set
                                        to 0 to use full original image resolution
                                        [default: 256]
    -f PIXELS --flow-filter=PIXELS      Filters out optical flow vectors that, when
                                        chaining forward and backward flows together, do
                                        not map back onto themselves within PIXELS. Set
                                        to 'off' to disable filtering. [default: 0.2]
    -i --interactive                    Enable interactive alignment of start and end
                                        frames
    -d --debug                          Enable more verbose logging and plot interme-
                                        diate results
    -o --outfile=OUTFILE                Save trimmed version of video in OUTFILE
    --ffmpeg-opts=OPTS                  Pass options OPTS (one quoted string) to ffmpeg,
                                        e.g. --ffmpeg-opts="-b:v 1000 -c:v h264 -an"
    -h --help                           Show this help text



DESCRIPTION:

Finds a loop in a repeating video, such as a concentric mosaic dataset, stored in
VIDEO_PATH.

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
from docopt import docopt
from schema import Schema, Use, And, Or, SchemaError
import ffmpeg
import os
from textwrap import dedent


# Set up custom logger
logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s\t%(message)s"))
logger.addHandler(handler)


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

    def __init__(
        self,
        video_path,
        start_frame_idx=0,
        duration_hint=None,
        *,
        resolution=256,
        flow_filter_threshold=0.2,
        match_brightness_range=None,
        debug=False,
        interactive=False,
    ):
        """Constructor

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
            interactive             — Enable interactive alignment of start and end
                                      frames
        """

        self.interactive = interactive
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

        self.match_brightness_range = match_brightness_range

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

        if resolution == 0:
            resolution = width
        self.resolution = (resolution, int(height / width * resolution))

        if duration_hint is None:
            self.end_frame_idx = self.video_duration - 1
        else:
            self.end_frame_idx = (
                min(self.video_duration, start_frame_idx + duration_hint) - 1
            )

        logger.info(f"Input loaded: video_duration={self.video_duration:.0f}")

        # Seek to start_frame_idx
        self.start_frame_idx = 0 if start_frame_idx is None else start_frame_idx
        self.start_frame = self._seek(self.start_frame_idx)

        # Initialise optical flow algorithm
        self.flow_algo = cv2.optflow.createOptFlow_Farneback()
        self.flow_filter_threshold = flow_filter_threshold

        # Determine looping direction
        self.loop_direction, self.vertical = self._find_video_direction()
        if self.vertical:
            logger.info("The camera appears to move vertically")
            logger.info(
                "Looping direction appears to be "
                f"{'down' if self.loop_direction == VideoLoopDirection.CW else 'up'}ward"
            )
        else:
            logger.info(f"Looping direction appears to be {self.loop_direction.name}")

        # Will be populated by find_closest_end_frame
        self.end_frames = None

    def _seek(self, frame_idx, downsample=True, grayscale=True, normalise=True):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = self.video.read()
        if not success:
            logger.error(f"Cannot read frame {frame_idx}")

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if downsample:
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
        if normalise:
            frame = cv2.normalize(frame.astype(float), None)

        return frame

    def _compute_pixel_difference(self, other_frame):
        """Compute mean absulute pixel difference between start_frame and other_frame"""
        return np.abs(self.start_frame - other_frame).mean()

    def _find_video_direction(self, frame1=None, frame2=None):
        """Determine the direction the video is spinning in between two frames

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
                verbose=self.debug,
            ).filled()

        x_flow = np.nanmedian(flow_forward[..., 0])
        y_flow = np.nanmedian(flow_forward[..., 1])

        vertical_flow = np.abs(x_flow) < np.abs(y_flow)

        if np.nanmedian(flow_forward[..., int(vertical_flow)]) < 0:
            return VideoLoopDirection.CW, vertical_flow
        else:
            return VideoLoopDirection.CCW, vertical_flow

    def find_closest_end_frame(self, search_range=50):
        """Find frame most similar to start frame that still lies before it, and sets
        end_frame_idx and end_frames member variables where
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
        idx_from = max(1, self.end_frame_idx - search_range)
        idx_to = min(self.video_duration - 2, self.end_frame_idx + search_range)
        end_frame_range = np.arange(idx_from, idx_to + 1)

        # Iterate over video with 3-frame window, searching for closest match
        prev_frame = None
        curr_frame = self._seek(idx_from - 1)
        next_frame = self._seek(idx_from)
        min_diff = np.inf
        min_idx = idx_from
        min_frames = tuple()  # 3 frames centered on current minimum
        mads = np.empty_like(end_frame_range, dtype=float)
        self.end_frame_cache = []
        for i in range(len(end_frame_range)):
            # Read new frame
            success, frame = self.video.read()
            if not success:
                msg = f"Failed to read frame {end_frame_range[i]}"
                logger.fatal(msg)
                raise RuntimeError(msg)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
            frame = cv2.normalize(frame.astype(float), None)

            # Shift frames along
            prev_frame = curr_frame
            curr_frame = next_frame
            next_frame = frame

            # Keep frames for interactive mode
            if self.interactive:
                self.end_frame_cache.append(curr_frame)

            # Test for minimum MAD
            mad = self._compute_pixel_difference(curr_frame)
            if self.debug or self.interactive:
                mads[i] = mad
            if mad and mad < min_diff:
                min_diff = mad
                min_idx = end_frame_range[i]
                min_frames = prev_frame, curr_frame, next_frame

        if self.loop_direction == self._find_video_direction(
            min_frames[1], self.start_frame
        ):
            self.end_frames = [min_frames[1], min_frames[2]]
            self.end_frame_idx = min_idx
        else:
            self.end_frames = [min_frames[0], min_frames[1]]
            self.end_frame_idx = min_idx - 1

        if self.debug | self.interactive:
            self._plot_dissimilarity(
                end_frame_range, mads, "Mean absolute pixel difference"
            )

        return self.start_frame_idx, self.end_frame_idx

    def match_brightness(self):
        """Search around start_frame_idx and end_frame_idx for the best match in
        brighness"""

        # Set search range as ±match_brightness_range, truncated by video length
        search_range = range(
            max(1, self.start_frame_idx - self.match_brightness_range)
            - self.start_frame_idx,
            min(
                self.video_duration - 1,
                self.end_frame_idx + self.match_brightness_range + 1,
            )
            - self.end_frame_idx,
        )

        # Compute average brightness in neighbourhood of start frame
        start_brightness = np.empty(len(search_range))
        frame = self._seek(
            self.start_frame_idx + search_range[0],
            downsample=False,
            grayscale=True,
            normalise=False,
        )
        start_brightness[0] = np.mean(frame)
        for i in range(1, len(search_range)):
            success, frame = self.video.read()
            if not success:
                msg = f"Failed to read frame {search_range[i]}"
                logger.fatal(msg)
                raise RuntimeError(msg)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            start_brightness[i] = np.mean(frame)

        # Compute average brightness in neighbourhood of end frame
        end_brightness = np.empty(len(search_range))
        frame = self._seek(
            self.end_frame_idx + search_range[0],
            downsample=False,
            grayscale=True,
            normalise=False,
        )
        end_brightness[0] = np.mean(frame)
        for i in range(1, len(search_range)):
            success, frame = self.video.read()
            if not success:
                msg = f"Failed to read frame {search_range[i]}"
                logger.fatal(msg)
                raise RuntimeError(msg)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            end_brightness[i] = np.mean(frame)

        # Exhausive search for global minimum of pairwise brightness difference
        brightness_difference = np.abs(start_brightness - end_brightness)
        min_idx = np.argmin(brightness_difference)

        self.start_frame_idx = self.start_frame_idx + search_range[min_idx]
        self.start_frame = self._seek(self.start_frame_idx)
        self.end_frame_idx = self.end_frame_idx + search_range[min_idx]

        logger.info(
            f"Frame pair closest in brightness: {self.start_frame_idx}, {self.end_frame_idx}"
        )

        if self.interactive:
            old_end_frame_idx = self.end_frame_idx
            self._plot_dissimilarity(
                self.end_frame_idx + search_range - search_range[min_idx],
                brightness_difference,
                "Absolute average brightness difference",
                False,
            )
            self.start_frame_idx += self.end_frame_idx - old_end_frame_idx
            self.start_frame = self._seek(self.start_frame_idx)

        return self.start_frame_idx, self.end_frame_idx

    def _plot_dissimilarity(
        self, end_frame_range, y_values, y_label, show_frame_diff=True
    ):
        """Plot mean absolute difference of pixels between two frames"""
        fig = plt.figure("Dissimilarity with start frame", figsize=(15, 7))
        ax = fig.subplots(1, 2) if show_frame_diff else [plt.axes()]
        curve = ax[0].plot(end_frame_range, y_values)
        marker = ax[0].plot(
            self.end_frame_idx,
            y_values[self.end_frame_idx - end_frame_range[0]],
            "r.",
        )
        ax[0].set_title(f"Start frame idx: {self.start_frame_idx}")
        ax[0].set_xlabel(f"end frame index: {self.end_frame_idx}")
        ax[0].set_ylabel(y_label)

        if show_frame_diff:
            im = ax[1].imshow(np.abs(self.start_frame - self.end_frames[0]), cmap="jet")
            plt.colorbar(im)

        if self.interactive:
            ax[0].set_title(
                f"Start frame idx: {self.start_frame_idx}\n"
                "Adjust with Ctrl(+Shift)+Left/Right"
            )
            ax[0].set_xlabel(
                f"end frame index: {self.end_frame_idx}\n"
                "Adjust with (Shift+)Left/Right"
            )
            if show_frame_diff:
                ax[1].imshow(np.abs(self.start_frame - self.end_frames[0]), cmap="jet")

            def key_handler(event):
                if event.key == "left":
                    self.end_frame_idx -= 1
                elif event.key == "shift+left":
                    self.end_frame_idx -= 10
                elif event.key == "right":
                    self.end_frame_idx += 1
                elif event.key == "shift+right":
                    self.end_frame_idx += 10

                elif event.key == "ctrl+left":
                    self.start_frame_idx -= 1
                elif event.key == "shift+ctrl+left":
                    self.start_frame_idx -= 10
                elif event.key == "ctrl+right":
                    self.start_frame_idx += 1
                elif event.key == "shift+ctrl+right":
                    self.start_frame_idx += 10

                elif event.key in ["enter", "escape"]:
                    plt.close()
                else:
                    return

                if "ctrl" in event.key:
                    self.start_frame_idx %= self.video_duration
                    self.start_frame = self._seek(self.start_frame_idx)
                    for i, frame in enumerate(self.end_frame_cache):
                        y_values[i] = self._compute_pixel_difference(frame)
                    curve[0].set_ydata(y_values)
                    ax[0].set_title(
                        f"Start frame idx: {self.start_frame_idx}\n"
                        "Adjust with Ctrl(+Shift)+Left/Right"
                    )
                else:
                    self.end_frame_idx = np.clip(
                        self.end_frame_idx, end_frame_range[0], end_frame_range[-1]
                    )
                    self.end_frames = self.end_frame_cache[
                        self.end_frame_idx
                        - end_frame_range[0] : self.end_frame_idx
                        - end_frame_range[0]
                        + 2
                    ]
                    ax[0].set_xlabel(
                        f"end frame index: {self.end_frame_idx}\n"
                        "Adjust with (Shift+)Left/Right"
                    )
                marker[0].set_data(
                    self.end_frame_idx,
                    y_values[self.end_frame_idx - end_frame_range[0]],
                )
                if show_frame_diff:
                    ax[1].imshow(
                        np.abs(self.start_frame - self.end_frames[0]), cmap="jet"
                    )
                fig.canvas.draw()

            fig.canvas.mpl_connect("key_press_event", key_handler)
            plt.show()

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
            verbose     — Show intermediate results
        Returns:
            A masked_array the same size as fwd_flow with inconsistent flow values masked
            out
        """
        height, width, depth = fwd_flow.shape

        if bwd_flow.shape != (height, width, depth) or depth != 2:
            raise RuntimeError(
                "Both input flows must have the same size and have 2 channels"
            )

        fwd_flow = np.ma.masked_array(fwd_flow, copy=True, fill_value=np.nan)

        img_coords_x, img_coords_y = np.meshgrid(np.arange(width), np.arange(height))
        img_coords = np.dstack((img_coords_x, img_coords_y)).astype(np.float32)
        coords_in_next = img_coords + fwd_flow
        coords_in_prev = (
            cv2.remap(
                bwd_flow,
                coords_in_next[..., 0],
                coords_in_next[..., 1],
                cv2.INTER_CUBIC,
                None,
            )
            + coords_in_next
        )
        error = np.linalg.norm(coords_in_prev - img_coords, axis=-1)
        if verbose:
            plt.figure("Histogram of optical flow relocalisation error")
            plt.hist(error.ravel(), bins=100, range=[0, 2])
            plt.xlabel("deviation in pixels")

        fwd_flow.mask = error > threshold

        if fwd_flow.mask.mean() > 0.5:
            logger.warning(
                "More than 50% of optical flow vectors have been filtered out. "
                "Consider increasing --flow-filter threshold"
            )

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
            logger.fatal(msg)
            raise RuntimeError(msg)

        # Compute optical flows 0→(N-1) and 0→N which should point in opposite
        # directions
        flows = [
            self.flow_algo.calc(self.start_frame, self.end_frames[0], None),
            self.flow_algo.calc(self.start_frame, self.end_frames[1], None),
        ]

        if self.flow_filter_threshold is not None:
            bwd_flows = [
                self.flow_algo.calc(self.end_frames[0], self.start_frame, None),
                self.flow_algo.calc(self.end_frames[1], self.start_frame, None),
            ]
            flows = [
                self.filter_optical_flow(
                    flows[i],
                    bwd_flows[i],
                    self.flow_filter_threshold,
                    verbose=self.debug,
                ).filled()
                for i in range(2)
            ]

        # We are only interested in the horizontal components
        flow_magnitudes = [np.abs(f[..., int(self.vertical)]) for f in flows]
        flow_magnitude_sum = sum(flow_magnitudes)

        full_frame_count = self.end_frame_idx - self.start_frame_idx
        fractional_frame_count = np.nanmedian(
            flow_magnitudes[0][flow_magnitude_sum != 0]
            / flow_magnitude_sum[flow_magnitude_sum != 0]
        )
        logger.info(
            f"Frame {self.start_frame_idx} lies at {100*fractional_frame_count:.0f}%"
            f" between frames {self.end_frame_idx} and {self.end_frame_idx + 1}"
        )
        if self.debug:
            plt.figure("Relative flow from end to start frame")
            plt.imshow(flow_magnitudes[0] / flow_magnitude_sum)
            plt.colorbar()
            plt.figure("Histogram of relative flow measurements")
            relative_flow_magnitude = flow_magnitudes[0] / flow_magnitude_sum
            plt.hist(
                relative_flow_magnitude[~np.isnan(relative_flow_magnitude)], bins=100
            )
            plt.xlabel(
                f"Relative position of frame {self.start_frame_idx} "
                f"between frames {self.end_frame_idx} and {self.end_frame_idx + 1}"
            )

        return full_frame_count / (full_frame_count + fractional_frame_count)

    @staticmethod
    def trim_video(in_filepath, from_idx, to_idx, out_filepath, ffmpeg_options):
        """Trim input video to [from_idx, to_idx], both inclusive"""
        (
            ffmpeg.input(in_filepath)
            .trim(start_frame=from_idx, end_frame=to_idx + 1)
            .setpts("PTS-STARTPTS")
            .output(out_filepath, **ffmpeg_options)
            .run()
        )


if __name__ == "__main__":

    opts = docopt(__doc__)
    schema = Schema(
        {
            "VIDEO_PATH": Use(str.strip),
            "START_FRAME_IDX": Or(None, And(Use(int), lambda f: f >= 0)),
            "DURATION_HINT": Or(None, And(Use(int), lambda d: d > 0)),
            "--range": And(Use(int), lambda r: r >= 0),
            "--match-brightness": And(Use(int), lambda r: r >= 0),
            "--width": And(Use(int), lambda w: w >= 0),
            "--flow-filter": Or(
                And(lambda f: f.lower().strip() == "off", Use(lambda f: None)),
                And(Use(float), lambda t: t >= 0),
                error="Valid --flow-filter values: 'off' or float > 0",
            ),
            "--outfile": Or(
                None,
                And(Use(str.strip), lambda f: not os.path.exists(f)),
                error="OUTFILE already exists",
            ),
            "--ffmpeg-opts": Use(
                lambda opts: {
                    kv[0]: " ".join(kv[1:]) if len(kv) > 1 else None
                    for opt in opts.split("-")
                    if len(opt) > 0
                    for kv in [opt.split()]
                }
                if opts
                else {}
            ),
            str: object,
        }
    )
    try:
        opts = schema.validate(opts)
    except SchemaError as e:
        exit(e)

    vlf = VideoLoopFinder(
        opts["VIDEO_PATH"],
        start_frame_idx=opts["START_FRAME_IDX"],
        duration_hint=opts["DURATION_HINT"],
        resolution=opts["--width"],
        flow_filter_threshold=opts["--flow-filter"],
        match_brightness_range=opts["--match-brightness"],
        debug=opts["--debug"],
        interactive=opts["--interactive"],
    )

    start_frame_idx, end_frame_idx = vlf.find_closest_end_frame(
        search_range=opts["--range"]
    )

    if opts["--match-brightness"] > 0:
        vlf.match_brightness()
        start_frame_idx, end_frame_idx = vlf.find_closest_end_frame(search_range=2)

    end_frame_position = vlf.localise_end_frame()

    print(
        dedent(
            f"""
        Loop detected
        Start frame: {start_frame_idx}
        End frame: {end_frame_idx}
        End frame position: {end_frame_position}
    """
        )
    )

    if opts["--outfile"]:
        logger.info(f"Exporting trimmed video to {opts['--outfile']}...")
        vlf.trim_video(
            opts["VIDEO_PATH"],
            start_frame_idx,
            end_frame_idx,
            opts["--outfile"],
            opts["--ffmpeg-opts"],
        )
        logger.info("...done")

    if opts["--debug"]:
        plt.show()
