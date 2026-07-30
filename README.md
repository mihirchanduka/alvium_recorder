# Alvium Recorder

A C++ application for recording video and metadata from one or multiple Allied Vision Alvium cameras using the Vimba X SDK and OpenCV.

## Features

- **Multi-Camera Support:** Interactively select one, multiple, or all available cameras for simultaneous recording.
- **Configurable Settings:** Set custom exposure times and recording durations per session.
- **Rich Output:** Records `.avi` video along with precise hardware/host timestamps (`.csv`), camera metadata (`.json`), and detailed logs.
- **Live Preview:** View real-time feeds from all active cameras while recording.
- **Standalone Vimba X Integration:** Auto-detects the local GenTL path and links embedded Vimba runtime paths, minimizing external system dependencies.

## Prerequisites

The project requires OpenCV, CMake, and a modern C++ compiler. On Fedora/RHEL-based systems, you can install the dependencies using:

```bash
sudo dnf install opencv-devel cmake gcc-c++
```

*(Note: The Vimba X SDK and GenTL transport layers are bundled in the `VimbaX_2026-1` directory.)*

## Build

From the project root, configure and build the project using CMake:

```bash
cmake -S . -B build
cmake --build build -j4
```

## Run

Execute the compiled binary from the project root:

```bash
./build/alvium_recorder
```

### Usage Instructions
1. **Camera Selection:** The app will list available cameras. Enter the indices of the cameras you want to use (e.g., `1,2`, `1-3`, or `all`).
2. **Parameters:** Enter the desired recording duration in seconds and the exposure time in microseconds.
3. **Recording:** Live preview windows will appear. The recording will run for the specified duration.
   - *To stop early:* Press `q` or `ESC` while focusing on any of the preview windows.
4. **Session Loop:** Once finished, the app will ask if you want to start another recording session.

## Output Structure

Recordings are saved in automatically generated directories (e.g., `recordings/rec_0001/`) relative to your working directory. A typical output folder contains:

- `session.log`: General session events and camera configuration details.
- `CamX.avi`: The recorded video file for camera X.
- `CamX_times.csv`: Frame-by-frame hardware and host timestamps.
- `CamX_meta.json`: Camera settings and metadata.
- `CamX.log`: Dedicated log file for camera X.

## Notes

- **GenTL Path:** The project auto-detects the local GenTL path from `VimbaX_2026-1/cti` if `GENICAM_GENTL64_PATH` is not already set in your environment.
- **Library Linking:** If the loader cannot find Vimba libraries when running, ensure you rebuild from the repository root so the correct embedded runtime library paths (RPATH) are applied.
