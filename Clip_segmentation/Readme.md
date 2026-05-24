# Video segmentation into speech-coherent clips

This notebook provides a solution for splitting long videos into shorter clips based on natural speech boundaries. Instead of cutting videos at arbitrary timestamps, it uses **speech transcription and segmentation** to ensure that each clip contains complete spoken sentences or thoughts, without cutting off a speaker mid-sentence.

## Features

- Splits videos into clips of a target duration (e.g., 20–40 seconds)
- Uses **OpenAI Whisper** to transcribe audio and detect natural segment boundaries
- Ensures that clips do not break in the middle of a spoken segment
- Processes an entire folder of videos automatically
- Exports clips as `.mp4` files
- Zips the output folder for easy download

## How It Works

1. **Extract audio** from the video file
2. **Transcribe audio** using Whisper (small model by default)
3. **Group transcript segments** into clips respecting the max duration
4. **Generate video clips** from the original video using start/end times of each group
5. **Skip clips** shorter than the minimum allowed duration
6. **Repeat** for all videos in the input folder

## Requirements

The notebook installs the following dependency automatically:

```bash
git+https://github.com/openai/whisper.git
```
Additionally, the following Python libraries are used:

- moviepy

- numpy

- whisper

- IPython

## Usage
1. Prepare Input Videos
   
Place your videos in a folder (e.g., /kaggle/input/datasets/mariaspasyuk/videos-part4/downloads4).
Supported formats: .mp4, .avi, .mov, .mkv, .webm, .flv, .m4v.

3. Run the Processing Function
```bash
process_videos_folder(
    input_folder="/path/to/videos",
    output_folder="/path/to/output",
    min_duration=20,   # minimum clip duration in seconds
    max_duration=35    # maximum clip duration in seconds
)
```
4. Download Results

The notebook automatically zips the output folder and provides a clickable download link.
