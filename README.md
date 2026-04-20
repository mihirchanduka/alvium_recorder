# alvium_recorder

```bash
sudo dnf install opencv-devel cmake gcc-c++
```

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build -j4
```

## Run

```bash
./build/alvium_recorder
```

## Notes

- The project auto-detects the local GenTL path from `VimbaX_2026-1/cti` if `GENICAM_GENTL64_PATH` is not already set.
- If the loader cannot find Vimba libraries, rebuild from this repo root so the embedded runtime paths are applied.

