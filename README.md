# reactions-generator

## How to run

```
poetry install
poetry run main --help
```

## Dev setup

This project is built using poetry. We also use ruff for formatting and linting, as well as pyright for type inference.

## Previous rendering approaches

1. Arange reaction via Pillow.
   - Saving frames as images (super slow).
   - Pyav/imageio-pyav - significantly slower.
   - Read and write using moviepy - the fastest option.
2. Compose reaction via moviepy - significantly slower than via Pillow.
3. Compose reaction using ffmpeg - the fastest option by far, current approach.
