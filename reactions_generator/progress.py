"""Submission progress for terminal sessions and Docker logs."""

import os
import sys
from contextlib import ExitStack
from time import monotonic
from typing import TYPE_CHECKING, Never, Self, TextIO, cast

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from reactions_generator.scheduler import SubmissionProgress

if TYPE_CHECKING:
    BaseTqdm = tqdm[Never]
else:
    BaseTqdm = tqdm


def plain_progress(file: TextIO) -> bool:
    return os.environ.get("REACTIONS_PROGRESS") == "plain" or not file.isatty()


def write_progress_message(message: str) -> None:
    """Keep status messages on separate lines without disrupting a live bar."""
    if plain_progress(sys.stderr):
        print(message, file=sys.stderr, flush=True)
    else:
        tqdm.write(message, file=sys.stderr)


class ServerTqdm(BaseTqdm):
    """Use tqdm's formatting and throttling with newline output in log mode."""

    last_display = 0.0

    def display(self, msg: str | None = None, pos: int | None = None) -> None:
        output = cast(TextIO, self.fp)
        if plain_progress(output):
            if msg == "":
                return
            output.write(f"{str(self) if msg is None else msg}\n")
            output.flush()
        else:
            super().display(msg, pos)
        self.last_display = monotonic()


class RenderProgress:
    """Report queue snapshots through one terminal bar or periodic log lines."""

    def __init__(self, file: TextIO | None = None) -> None:
        self._file = file if file is not None else sys.stderr
        self._plain = plain_progress(self._file)
        self._contexts = ExitStack()
        self._bar: ServerTqdm | None = None
        self._last_snapshot: SubmissionProgress | None = None
        self.completed = 0
        self.total = 0
        self.failed = 0

    def __enter__(self) -> Self:
        if not self._plain:
            self._contexts.enter_context(logging_redirect_tqdm())
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            if self._bar is not None:
                self._bar.close()
        finally:
            self._contexts.close()

    def report(self, snapshot: SubmissionProgress) -> None:
        self.completed = snapshot.succeeded + snapshot.failed
        self.total = self.completed + snapshot.remaining
        self.failed = snapshot.failed
        if self._bar is None:
            self._bar = ServerTqdm(
                total=self.total,
                initial=self.completed,
                desc="Rendering submissions",
                unit="it",
                file=self._file,
                mininterval=30 if self._plain else 0.1,
                miniters=1,
                dynamic_ncols=not self._plain,
                ascii=self._plain,
                leave=not self._plain,
                postfix=f"failed={self.failed}",
                bar_format=(
                    "{desc}: {bar:16} {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
            )
        else:
            total_changed = self._bar.total != self.total
            self._bar.total = self.total
            self._bar.set_postfix_str(f"failed={self.failed}", refresh=False)
            advance = self.completed - self._bar.n
            if advance < 0:
                self._bar.initial += advance
            displayed = self._bar.update(advance)
            drained = snapshot.remaining == 0 and snapshot != self._last_snapshot
            periodic_log = self._plain and monotonic() - self._bar.last_display >= 30
            failed_changed = (
                self._last_snapshot is not None
                and self.failed != self._last_snapshot.failed
            )
            if not displayed and (
                total_changed
                or drained
                or periodic_log
                or advance < 0
                or failed_changed
            ):
                self._bar.refresh()
        self._last_snapshot = snapshot
