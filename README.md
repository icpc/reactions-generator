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

## Continuous rendering and priorities

```sh
poetry run main all https://overlayer.example --poll-interval 2
```

`all` polls `/reactions/runs.json` in the background while rendering one video at
a time. `--poll-interval` is the delay between polls in seconds (default: 2).
New submissions can therefore enter the queue during a render. Each worker
selects its next render using this priority order:

1. First-to-solve submissions (`isFirstToSolve`).
2. Other accepted submissions (`isAccepted`).
3. Remaining judged submissions.

These fields come from the [live-v3 reactions exporter](https://github.com/icpc/live-v3/blob/73195bdc4becdcd32db321d670a050dd0ebff076/src/cds-converter/src/main/kotlin/org/icpclive/converter/export/reactions/ReactionsExporter.kt).
Omitted `isFirstToSolve` means false; for IOI contests the exporter maps first-best
runs to that field. Within each priority, submissions retain their order of first
discovery. Updated verdicts change the priority of waiting submissions; an
encoding already in progress finishes before another begins. A steady stream of
higher-priority submissions can delay lower-priority ones.

Runs marked `IN_PROGRESS` wait until judged. Each successful poll refreshes the
waiting queue, removes runs no longer present, and avoids duplicating a running
submission. Existing output videos are skipped, and `--total-workers` /
`--worker-id` retain deterministic SHA-256 partitioning across worker processes.

Polling errors are logged and retried after the poll interval. A failed render
writes `<id>.err` in the output directory and can be retried on a later poll;
existing videos are preserved on failure. The queue is held in memory and rebuilt
from the API and existing output files after restart.

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
- `scheduler.py` maintains the waiting queue, priorities, and in-flight entries.
- `card.py`, `text_layout.py`, and the supporting modules draw animated cards.
- `tests/` contains regression tests using Python's standard-library `unittest`.

## Follow-up work

- Investigate moving each Python rendering job into a worker subprocess. Reproduce
  the earlier FFmpeg subprocess issues and check raw-frame pipe handling, stderr
  draining, error propagation, interruption, and child-process cleanup before
  changing the execution model. FFmpeg itself already runs as a subprocess.

## Previous rendering approaches

1. Arange reaction via Pillow.
   - Saving frames as images (super slow).
   - Pyav/imageio-pyav - significantly slower.
   - Read and write using moviepy - the fastest option.
2. Compose reaction via moviepy - significantly slower than via Pillow.
3. Compose reaction using ffmpeg - the fastest option by far, current approach.
