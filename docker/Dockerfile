# syntax=docker/dockerfile:1
#
# Two stages. The builder collects wheels; the runtime gets the wheels and the
# source and nothing else — no pip cache, no build tooling.
#
# Python 3.13, matching .python-version. Every pinned package ships a wheel
# for it, so the builder needs no compiler.

# ---------------------------------------------------------------- builder ---
FROM python:3.13-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /wheels

COPY requirements-core.txt requirements-dev.txt ./
# No `|| true`: a failed dependency install must fail the build, not ship an
# image that cannot import its own code.
RUN pip wheel --wheel-dir /wheels -r requirements-core.txt -r requirements-dev.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DATA_DIR=/data

WORKDIR /app

# libgomp is LightGBM's OpenMP runtime — without it `import lightgbm` fails at
# load time with an error that looks nothing like a missing system package.
# (The macOS equivalent is `brew install libomp`; see the README.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements-core.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
      -r requirements-core.txt -r requirements-dev.txt \
 && rm -rf /wheels

COPY . /app

# Non-root: nothing here needs to write outside the mounted data directory.
# /data is the mount point for the named volume. Docker initialises a named
# volume from the image's directory, ownership included, so creating it here
# as `forecast` is what makes the volume writable by a non-root container.
RUN useradd --create-home --uid 1000 forecast \
 && mkdir -p /app/artifacts /data \
 && chown -R forecast:forecast /app /data
USER forecast

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=6 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
