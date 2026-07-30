# x-team-submission

Submission for x-team problem: route a customer support ticket to one of four teams.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv.lock` is committed, so
the environment below is the exact one every number in this README was produced on.

```bash
uv sync            # creates .venv and installs the locked dependency set
uv run pytest      # run the tests
```

Anything prefixed with `uv run` executes inside that environment. No manual activation needed.

## How to read this repo

The commit history is the argument. Each commit takes one assumption about the data,
tests it, and records what came back in `results/`. The README grows the same way, so
reading it top to bottom is the same path as reading the commits in order.
