# YouTube High-Quality Video Downloader

A Python script that downloads YouTube videos with guaranteed quality (1080p, 1440p, 2160p, etc.), automatically handling codec limitations by converting VP9/AV1 to H.264 when necessary.

## Features

- Checks available video formats before downloading
- Prioritizes H.264 (AVC) for maximum compatibility
- Automatically converts VP9 or AV1 to H.264 using FFmpeg when H.264 is unavailable at target quality
- Displays available resolutions per codec (H.264, VP9, AV1)
- Batch download multiple videos
- Uses `yt-dlp` 

## Requirements

- Python 3.6+
- [FFmpeg](https://ffmpeg.org/) installed and accessible in system PATH
- Firefox browser (or other for cookies; can be modified)

## Installation

1. Clone or download this script.

2. Install the required Python package:
```bash
pip install yt-dlp
```
## Usage
Basic single video download
```bash
from link_download import download_guaranteed_quality

download_guaranteed_quality(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    output_path="./downloads",
    target_quality="1080")
```
Check available formats without downloading
```bash
from link_download import download_multiple_videos_guaranteed

urls = [
    "https://www.youtube.com/watch?v=VIDEO_1",
    "https://www.youtube.com/watch?v=VIDEO_2"
]

stats = download_multiple_videos_guaranteed(
    urls=urls,
    output_path="./high_quality_videos",
    target_quality="1080"
)
```
## How it works
- Analyzes available formats (H.264, VP9, AV1)

If H.264 at target quality exists → downloads directly as .mp4

If not → downloads best quality (VP9/AV1) + audio → converts video stream to H.264 with FFmpeg

Merges video and audio into a single .mp4 file

- Output format
  
Always produces .mp4 container

- Notes
  
Cookies from Firefox are used to avoid age restrictions / rate limits (change browser in code if needed)

Temporary files are automatically cleaned up after conversion

If target quality is not available at all, the script will abort without downloading

- Limitations
  
Requires FFmpeg for conversion (direct H.264 downloads work without FFmpeg)

Conversion takes time and CPU resources (depends on video length/resolution)
