# reactions-generator

See our reaction videos in action: https://news.icpc.global/reactions/

## How to run

Use Python 3.12 and install the system `ffmpeg` and `ffprobe` executables. Run
commands from the repository root so the default media and font paths resolve.

```sh
poetry env use python3.12
poetry install
poetry run main --help
```

The CLI provides `card` for animated cards, `reaction` for manual reaction videos,
`single` for one overlayer submission, and `all` for continuous rendering. Run
`poetry run main <command> --help` for options.

## Dev setup

Poetry manages dependencies, Ruff checks formatting, and Pyright checks types.
Run the same validation as GitHub Actions:

```sh
poetry run ruff format --check .
poetry run pyright
poetry run python -m unittest discover -s tests
```

The unit tests use local fixtures and mocks; they need no live overlayer or
FFmpeg process. For changes to composition or encoding, also render a short local
video and inspect its video/audio streams with `ffprobe`. Ruff linting is available
separately with `poetry run ruff check .`.

## Code organization

- `cli.py` registers commands and starts Typer.
- `media.py` loads images and probes media metadata.
- `rendering.py` composes reaction videos and encodes card frames with FFmpeg.
- `overlayer.py` parses API data into typed submissions and renders single runs.
- `worker.py` polls submissions, partitions work, and records rendering errors.
- `card.py`, `text_layout.py`, and the supporting modules draw animated cards.
- `tests/` contains regression tests using Python's standard-library `unittest`.

## Previous rendering approaches

1. Arange reaction via Pillow.
   - Saving frames as images (super slow).
   - Pyav/imageio-pyav - significantly slower.
   - Read and write using moviepy - the fastest option.
2. Compose reaction via moviepy - significantly slower than via Pillow.
3. Compose reaction using ffmpeg - the fastest option by far, current approach.
