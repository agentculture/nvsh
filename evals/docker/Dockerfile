# Autonomous driver for the DeepEval release gate (issue 64, deviation d2).
#
# Runs `python -m evals.tool_jev drive` until the run is complete. Built from
# the repository root (see compose.yaml); holds no keys and no case data:
# keys arrive as environment variables at `docker compose up` time
# (`grant run --inject ...`), case data and the run directory are mounts.
FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    DEEPEVAL_DISABLE_DOTENV=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first, so a code change does not re-resolve them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group evals --no-install-project

# Only what the gate imports: the nvsh package, the evals suite and the
# fine-tune scripts it loads by path (metrics, gate, measure, scorer).
COPY nvsh ./nvsh
COPY evals ./evals
COPY scripts/lfm-finetune ./scripts/lfm-finetune
RUN uv sync --frozen --no-dev --group evals

# deepeval writes .deepeval/ into the working directory: keep it in the run dir.
# The venv's python runs directly: the image is never re-synced at run time
# (the container user cannot write /opt/venv, and nothing needs to).
WORKDIR /run-dir
ENTRYPOINT ["/opt/venv/bin/python", "-m", "evals.tool_jev"]
CMD ["drive"]
