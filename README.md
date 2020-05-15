# video-loop-finder
Tool to find matching start and end points in a looping video, e.g. in a concentric mosaic light field dataset

## Setup
Install depencencies listed in `environment.yml`.
If Anaconda is set up, simply run:
```bash
conda env create -f environment.yml
```

## Usage
Check `video_loop_finder.py --help` for possible options.

<pre>
Video Loop Finder

USAGE:
    video_loop_finder.py [options] VIDEO_PATH [START_FRAME_IDX [DURATION_HINT]]

ARGUMENTS:
    VIDEO_PATH          Path to a video file or printf-style escaped path to image
                        sequence, e.g. '/path/to/image%04d.png'
    START_FRAME_IDX     Index of first frame of loop [default: 0]
    DURATION_HINT       Estimated duration of loop in frames [default: video duration]

OPTIONS:
    -r RANGE --range=RANGE          Search for end frame ±RANGE frames around
                                    START_FRAME + DURATION_HINT [default: 50]
    -w WIDTH --width=WIDTH          Image width in pixels used in computations. Set to 0
                                    to use full original image resolution [default: 256]
    -f PIXELS --flow-filter=PIXELS  Filters out optical flow vectors that,
                                    when chaining forward and backward flows together,
                                    do not map back onto themselves within PIXELS. Set
                                    to 'off' to disable filtering. [default: 0.2]
    -i --interactive                Enable interactive alignment of start and end frames
    -d --debug                      Enable more verbose logging and plot intermediate
                                    results
    -o --outfile=OUTFILE            Save trimmed version of video in OUTFILE
    --ffmpeg-opts=OPTS              Pass options OPTS (one quoted string) to ffmpeg,
                                    e.g. --ffmpeg-opts="-b:v 1000 -c:v h264 -an"
    -h --help                       Show this help text



DESCRIPTION:

Finds a loop in a repeating video, such as a concentric mosaic dataset, stored in
VIDEO_PATH.

This script will find the best matching frame pair in terms of lowest sum of absolute
pixel differences and localise the end frame relative to the actual beginning/end of the
loop.

For example, if in a concentric mosaic video, the first frame is assumed at 0° and the
closest end frame is found at 359.1°, then the relative position of the latter is
359.1°/360° = 99.75%.
</pre>
