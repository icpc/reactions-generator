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

`all` polls `/reactions/runs.json` in a dedicated thread while rendering one
video at a time by default. Set `--total-workers` to control local concurrency.
A fixed pool of renderer threads takes videos from the priority queue. Each
renderer finishes its video before taking the highest-priority waiting submission.
Renderers share one progress bar showing processed submissions out of the current
total, including failed attempts. New eligible submissions increase the total;
vanished waiting submissions decrease it. Routine status messages are suppressed.
When `--worker-id` is supplied, this process renders one partition and
`--total-workers` specifies the number of distributed partitions. `--poll-interval` is the delay between polls in seconds (default: 2).
The pool has one polling thread in addition to the render threads. The main thread
waits for shutdown. Renderers continue taking queued videos during API requests;
new submissions enter the queue on each poll. Shutdown stops dispatching new work
and waits for active jobs and the current API request (timeout: 30 seconds).
Each worker
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
submission. Existing output videos are skipped. With `--worker-id`, deterministic SHA-256
partitioning is retained across worker processes. Priority applies when each
render slot becomes available; active renders are never interrupted.

Polling errors are logged and retried after the poll interval. A failed render
writes `<id>.err` in the output directory and is skipped on later polls and after
restart. Delete its `.err` file to allow an intentional retry. Shutting down with
Ctrl+C or SIGTERM does not mark interrupted renders as failed. Existing videos
are preserved on failure. The queue is held in memory and rebuilt
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

Image and font helpers use bounded LRU caches (128 entries per cache). Each
completed task releases its card and resized logo; shared caches retain recent
entries for reuse without growing indefinitely. The bounds count entries, not bytes.

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


## Encoding and performance

Every FFmpeg render writes an info-level report to
`out/logs/<output-name>-<unique-id>.log`, including successful and interrupted
renders. Reports include FFmpeg's command, stream information, warnings, and
encoding statistics. Logs are retained separately from temporary output videos;
concurrent renders and retries use different filenames. Console progress behavior
is unchanged.

CPU H.264 encoding is the safe default: `libx264 -preset veryfast -crf 23`.
Hardware acceleration is **opt-in**; there is no automatic GPU selection or
silent fallback. Select a backend explicitly with `--hardware`. `--hwaccel`
controls input decoding separately and is not enabled automatically because
scaling and composition currently use software FFmpeg filters.

| Platform | `--hardware` | Encoder | Default speed setting |
| --- | --- | --- | --- |
| CPU-only (any OS) | `cpu` (default) | `libx264` | `veryfast` |
| NVIDIA (Linux/Windows) | `nvidia` | `h264_nvenc` | `p3` |
| Intel iGPU (Linux/Windows) | `intel` | `h264_qsv` | `veryfast` |
| AMD (Windows / supported Linux drivers) | `amd` | `h264_amf` | `speed` |
| Intel/AMD Linux via VAAPI | `vaapi` | `h264_vaapi` | driver-controlled |
| Apple Silicon / supported Intel Mac | `apple` | `h264_videotoolbox` | driver-controlled |

`--quality low|balanced|high` (default `balanced`) maps to these encoder-specific
parameters. The numbers are **not comparable in perceptual quality** across
encoders; Apple VideoToolbox uses a reversed quality direction.

| Encoder quality control | Low | Balanced | High |
| --- | ---: | ---: | ---: |
| x264 CRF | 26 | 23 | 18 |
| NVENC CQ | 27 | 24 | 19 |
| QSV global_quality | 27 | 24 | 19 |
| AMF CQP | 27 | 24 | 19 |
| VAAPI QP | 27 | 24 | 19 |
| VideoToolbox q:v | 45 | 60 | 75 |

Quality controls output fidelity/file size; `--preset` controls compression
speed/efficiency and is validated for the selected backend. `--vcodec` is an
advanced explicit encoder override, incompatible with non-CPU `--hardware`.
Unknown custom encoders are passed through without automatic quality mapping.

```sh
poetry run main reaction                                      # CPU default
poetry run main reaction --hardware nvidia --quality balanced
poetry run main reaction --hardware vaapi --vaapi-device /dev/dri/renderD128
poetry run main all https://overlayer.example --hardware intel --total-workers 2
poetry run main benchmark --hardware all --worker-counts 1,2,4,8
```

`benchmark` performs a short capability probe, then full application renders
(card generation, decoding, scaling, overlays, audio, encoding) using five-second clips from the local
sample media for each working backend and each requested worker count. Clips are
prepared once using stream copying, preserving source codecs. Preparation is
excluded from timing. Each batch renders `max(worker_counts)` videos by default,
so every worker count processes the same workload. For a longer measurement use
`--duration-seconds 15 --jobs-per-worker 2 --repetitions 3`. Stream-copy boundaries
can make clip duration slightly exceed the requested limit. Short benchmarks are
useful for a quick comparison; confirm close results with longer runs. It
writes a JSON report to `out/benchmarks/results.json`; it cleans temporary
videos and never touches production output. The fastest successful combination
is recommended based on completed videos per second, not guaranteed equivalent
perceptual quality. Start with counts `1,2,4` and extend if throughput improves.
In production, `all --total-workers N` runs N local render threads when
`--worker-id` is omitted; with an explicit `--worker-id` it retains distributed
partitioning semantics. The benchmark does not contact the overlayer.

GPU encoding requires an FFmpeg build with the encoder enabled and working
runtime drivers. NVIDIA in Docker additionally requires exposing the GPU through
the NVIDIA container runtime; VAAPI in Linux containers requires accessible
`/dev/dri/renderD*` devices and compatible drivers. The standard CPU Docker
image continues to work without GPUs. A compiled-in encoder is not proof that
the hardware or driver will accept encoding: use `benchmark` to verify.
