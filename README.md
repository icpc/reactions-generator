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
Each render downloads its remote webcam and screen media into a temporary
directory before probing or encoding. The files are removed when the render
ends, including after failures or interruptions.
Renderers share one Rich progress bar showing processed submissions, the current
total, failed attempts, submissions per second, and estimated time remaining.
New eligible submissions increase the total;
vanished waiting submissions decrease it. Each render prints a start message and,
on success, a finish message on separate lines without disrupting the bar.
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

Docker Compose defaults to plain progress output. The worker prints a summary
on the first poll, every 30 seconds, and when the queue drains. Follow
those summaries and the per-render messages with `docker compose logs -f reactions`.
The Compose command must run `all ...` to show submission progress.

To view the live bar in the service's terminal, select automatic progress mode
when recreating the container and attach directly.

```sh
REACTIONS_PROGRESS=auto docker compose up -d --force-recreate
docker compose attach reactions
```

Use Ctrl+P, Ctrl+Q to detach. Ctrl+C requests shutdown. Outside Compose, Rich
uses the live bar when stderr is a terminal. Set `REACTIONS_PROGRESS=plain` to
force periodic summaries. Compose runs the published image, so this behavior
requires an updated image.

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

CPU H.264 encoding is the safe default: `libx264 -preset veryfast -crf 26`.
Hardware acceleration is **opt-in**; there is no automatic GPU selection or
silent fallback. Select a backend explicitly with `--hardware`. `--hwaccel`
controls input decoding separately and is not enabled automatically because
scaling and composition use software FFmpeg filters in the existing profiles.
The opt-in `nvidia_full` profile below manages CUDA decoding and composition itself.

| Platform | `--hardware` | Encoder | Default speed setting |
| --- | --- | --- | --- |
| CPU-only (any OS) | `cpu` (default) | `libx264` | `veryfast` |
| NVIDIA (Linux/Windows) | `nvidia` | `h264_nvenc` | `p3` |
| NVIDIA CUDA composition | `nvidia_full` | `h264_nvenc` | `p5` |
| Intel iGPU (Linux/Windows) | `intel` | `h264_qsv` | `veryfast` |
| AMD (Windows / supported Linux drivers) | `amd` | `h264_amf` | `speed` |
| Intel/AMD Linux via VAAPI | `vaapi` | `h264_vaapi` | driver-controlled |
| Apple Silicon / supported Intel Mac | `apple` | `h264_videotoolbox` | driver-controlled |

`--quality very-low|low|balanced|high` (default `balanced`) uses quality-based encoding,
so simple cards and static screens can compress to small files.

| Encoder quality control | Very low | Low | Balanced | High |
| --- | ---: | ---: | ---: | ---: |
| x264 CRF | 34 | 29 | 26 | 21 |
| NVENC CQ (both paths) | 42 | 39 | 36 | 29 |
| QSV global_quality | 38 | 32 | 29 | 23 |
| AMF CQP | 38 | 32 | 29 | 23 |
| VAAPI CQP | 38 | 32 | 29 | 23 |
| VideoToolbox q:v | 1 | 5 | 10 | 30 |

These values are arbitrary starting points on each encoder's own quality scale.
Lower values increase quality except for VideoToolbox, whose scale is reversed.
They favor small reaction videos with static cards and screens. Actual sizes
and perceptual quality vary with content, device, and driver. AAC audio uses
96 kb/s. Use `--quality very-low` for smaller files with more visible compression,
or `--quality high` when preserving more detail matters.

On the bundled 33.3-second reaction, a Tesla V100 with FFmpeg 7.1.5 produced
NVIDIA output at 1.059 times CPU size and full CUDA output at 1.108 times CPU
size using balanced CQ 36 and CPU CRF 26. Apple q:v 10 produced 1.83 MB in a
separate macOS run. QSV, VAAPI, and AMD values have command-level validation
only. Check the resulting card text and webcam detail on representative footage.
QSV uses ICQ with lookahead disabled, and VAAPI uses explicit CQP for driver
compatibility.

Quality controls output fidelity/file size; `--preset` controls compression
speed/efficiency and is validated for the selected backend. `--vcodec` is an
advanced explicit encoder override, incompatible with non-CPU `--hardware`
except that `nvidia_full` also accepts an explicit `--vcodec h264_nvenc`.
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


### NVIDIA full-CUDA renderer

`--hardware nvidia_full` is an opt-in path for machines with an NVIDIA GPU and
CUDA-capable FFmpeg. It is distinct from `--hardware nvidia` (the regular NVIDIA
encoding profile): `nvidia_full` decodes moving webcam/screen video into CUDA
frames, resizes with `scale_cuda`, composites with `overlay_cuda`, and encodes
with `h264_nvenc`. The full canvas never round-trips through Python/CPU.
Pillow still generates the small RGBA animated card; FFmpeg converts/uploads
that card with YUVA420P alpha and mixes audio on CPU.

```bash
main reaction --hardware nvidia_full
main reaction --hardware nvidia_full --quality high --preset p5
main single OVERLAYER_URL RUN_ID --hardware nvidia_full
main all OVERLAYER_URL --hardware nvidia_full --total-workers 1
main benchmark --hardware all --worker-counts 1,2,4,8
```

Default `nvidia_full` encoding uses H.264 NVENC, preset P5, HQ tuning,
VBR, CQ 36, and two B-frames. Both NVIDIA paths share the quality mapping above.
All renderers resolve encoding through the same `EncoderConfig` and share
audio/output options and the FFmpeg process runner. The CUDA backend supplies
its own decode/scale/overlay graph to keep video frames on the GPU.
Explicit NVENC presets are `p1` through `p7`. The quality name `medium`
is retained as an alias for `balanced` on this backend.

The full CUDA path requires `scale_cuda` with its `format` option, needed to
produce YUV420P video for the alpha-card overlay. Check
`ffmpeg -hide_banner -h filter=scale_cuda`; if `format` is absent, upgrade the
CUDA-enabled FFmpeg build. Ordinary `--hardware nvidia` can still work on such a
build because it uses software composition.

Check `nvidia-smi`, `ffmpeg -encoders | grep h264_nvenc`, and
`ffmpeg -filters | grep -E 'scale_cuda|overlay_cuda|hwupload'`. Actual rendering
still needs a compatible driver, runtime, and input codec. The generic Dockerfile
and any plain Debian `ffmpeg` package cannot be assumed to work on the GPU.
For NVIDIA Docker execution, arrange a CUDA/NVENC-compatible FFmpeg build and
NVIDIA Container Toolkit GPU access. Explicit `nvidia_full` fails rather than
silently falling back to software rendering.

Start by benchmarking 1, 2 and 4 workers on your actual GPU and also test 8
workers if VRAM/session capacity allows. CUDA visual equivalence and throughput
must be confirmed on the real target GPU; they were not tested on the authoring
machine.

The standalone `card` command uses this profile’s NVENC settings with software
card input; CUDA composition applies to reaction renders.

Implementation references: [FFmpeg CUDA overlay](https://ffmpeg.org/doxygen/trunk/vf__overlay__cuda_8c_source.html)
and the [NVIDIA FFmpeg guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/ffmpeg-with-nvidia-gpu/).
