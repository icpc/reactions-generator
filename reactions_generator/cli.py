"""Command registration; rendering and API logic live in their own modules."""

import typer

from reactions_generator.overlayer import build_submission
from reactions_generator.rendering import render_card, render_reaction
from reactions_generator.worker import continuous_build_submission

app = typer.Typer(no_args_is_help=True)
app.command("card", help="Render a manual card.")(render_card)
app.command("reaction", help="Render a manual vertical reaction.")(render_reaction)
app.command("single", help="Render one submission from the overlayer.")(
    build_submission
)
app.command("all", help="Continuously render all submissions from the overlayer.")(
    continuous_build_submission
)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
