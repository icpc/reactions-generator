# Repository guidance

## Owner preferences

The owner has no strong attachment to the current project structure. Use your
judgment to improve organization when it helps the requested work. Lint,
formatting, and type-checking expectations come from GitHub Actions; keep those
checks passing, fix issues introduced by your changes, and explicitly report any
existing failures or blockers.

## Project and architecture

This is a Python CLI for generating ICPC contest reaction videos. Pillow renders
animated RGBA cards; FFmpeg composes the cards, webcam footage, screen recordings,
backgrounds, and audio. Keep video composition in FFmpeg rather than replacing it
with per-frame Python composition or intermediate image files; the README records
the performance reasons for this approach.

Read the files relevant to the task:

- `reactions_generator/cli.py`: registration of Typer commands (`card`,
  `reaction`, `single`, `all`) and the CLI entry point.
- `reactions_generator/media.py`: image loading and media metadata probing.
- `reactions_generator/rendering.py`: FFmpeg composition and card video encoding.
- `reactions_generator/overlayer.py`: typed submission parsing, API requests, and
  rendering of individual submissions.
- `reactions_generator/worker.py`: polling, deterministic partitioning, and
  per-submission error reporting.
- `reactions_generator/scheduler.py`: thread-safe priority scheduling and
  in-flight submission tracking.
- `reactions_generator/card.py`: immutable `Card` model and frame rendering.
- `reactions_generator/text_layout.py`, `utils.py`, and `interpolate.py`: text
  fitting, image geometry, and animation helpers.
- `reactions_generator/defaults.py`, `colors.py`, and `fonts.py`: shared defaults,
  palette, and cached font loading.
- `example/` and `fonts/nimbus/`: bundled media and fallback fonts. Preserve asset
  licenses and the audio attribution in `example/README.md`.
- `pyproject.toml`, `poetry.lock`, `Dockerfile`, `docker-compose.yml`, and
  `.github/workflows/`: dependencies, packaging, containers, and CI.
- `tests/`: regression tests using standard-library `unittest` and mocks.

## Setup and commands

Run commands from the repository root: default media and font paths are relative
to the working directory. CI and Docker use Python 3.12; use it for reproducible
validation even though the package permits newer Python versions.

```sh
poetry env use python3.12
poetry install
poetry run main --help
poetry run main reaction --help
```

Rendering also requires the system `ffmpeg` and `ffprobe` executables on `PATH`.
The `typed-ffmpeg` dependency supplies the Python `ffmpeg` module; avoid installing
the unrelated `ffmpeg` or `ffmpeg-python` packages alongside it.

Use Poetry for dependency changes and update `pyproject.toml` and `poetry.lock`
together. Do not hand-edit lockfile package versions or hashes.

## Implementation conventions

- Match the existing code and Ruff formatting. Preserve type annotations; Pyright
  runs in strict mode without global diagnostic suppressions. Prefer
  precise types at external-data boundaries over broad new suppressions.
- Keep Typer command names, flags, defaults, and output naming compatible unless
  the requested change intentionally alters them. Reflect user-facing changes in
  the README or CLI help.
- Preserve RGBA card frames and their dimensions when piping raw video to FFmpeg;
  the input stream declaration must match the bytes produced by Pillow/NumPy.
- Keep time units explicit: contest time and audio delay are milliseconds,
  media duration is seconds, and animation positions are frame indices.
- Geometry helpers use `(left, top, right, bottom)` boxes, while layout defaults
  use `(x, y, width, height)`. Convert deliberately when moving between them.
- Preserve the custom-font-first, bundled-Nimbus-fallback behavior. Cached fonts
  and rendered images are reused; copy images before modifying cached results.
- For rendering changes, preserve child-process cleanup, temporary-output
  handling, and replacement of the destination only after successful encoding.
- For overlayer changes, preserve deterministic SHA-256 worker partitioning,
  skipping existing submissions unless overwrite is requested, and per-submission
  error handling. `all` polls in the background while encoding one video at a time.
- Preserve queue priority: first-to-solve, accepted, then other judged runs, using
  `isFirstToSolve` and `isAccepted` from the live-v3 short-run API. Missing
  `isFirstToSolve` means false. Keep first-discovery FIFO order within a priority,
  refresh waiting priorities on polls, and never preempt an active encode.
- Keep `IN_PROGRESS` runs out of the render queue until judged. Poll snapshots
  remove vanished waiting runs and deduplicate in-flight work. Poll failures
  retry after `--poll-interval`; failed renders record `.err` files and may retry
  on later polls. Use bounded fixtures and synchronization events for queue tests.

## Validation

For Python changes, run the checks defined in
`.github/workflows/format-and-types.yml`, which is the source of truth for the
owner's required validation:

```sh
poetry run ruff format --check .
poetry run pyright
poetry run python -m unittest discover -s tests
```

Use `poetry run ruff format <changed-files>` to apply formatting. Ruff linting
(`poetry run ruff check .`) is an additional useful check, but is not currently
required by Actions. Unit tests use fixtures and mocks without live network
requests or FFmpeg processes; keep those checks deterministic.

Validate the behavior touched by the change. Use small local fixtures or short
renders and write generated media to ignored `out/` or a temporary directory.
For visual changes, inspect frames before, during, and after the verdict
transition, with accepted and rejected outcomes and long text. For composition
changes, check the affected orientations, stream dimensions, timing, and audio
with `ffprobe` and playback. Mock overlayer responses for repeatable checks.

Documentation-only changes need a content/diff review, not a video render.
Report the checks actually run and any missing tools, assets, or dependencies
that prevented validation; distinguish existing failures from regressions.

## Containers and existing pitfalls

- `docker-compose.yml` runs the published GHCR `latest` image; it has no local
  build configuration. `docker compose up` therefore does not test source edits.
  To test a local image, build it explicitly, for example:
  `docker build -t reactions-generator:local .`.
- The Docker entrypoint is `main`; pass CLI arguments after the image name.
  Container dependencies exclude the development tools. Keep required runtime
  assets included in the Docker build when changing their locations.
- The horizontal background default points to `example/background-moscow.png`,
  which is not tracked. Supply an existing background for horizontal validation.
- Local custom fonts, root Helvetica files, `out/`, virtual environments, and
  `research.ipynb` are ignored. Do not rely on them being available in a fresh
  checkout or commit generated media, caches, or machine-specific files.

Keep changes focused on the requested task. Update this file when its commands
or project-specific guidance become stale.
