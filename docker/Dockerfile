# syntax=docker/dockerfile:1
#
# Two stages. The builder compiles wheels; the runtime gets the wheels and the
# source and nothing else — no compiler, no pip cache, no dev headers. The
# previous single-stage image shipped its whole build environment.
#
# Python 3.11: the pinned scientific stack (numpy 1.26, pandas 2.1) has no
# wheels for 3.13+, and building it from source in a container is not a thing
# anyone should wait for.

# ---------------------------------------------------------------- builder ---
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /wheels

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements-dev.txt ./
# No `|| true`. A failed dependency install used to produce a *successful*
# build and an image that could not import its own code — the failure was
# hidden until runtime, which is the worst place to discover it.
RUN pip wheel --wheel-dir /wheels -r requirements-core.txt -r requirements-dev.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.11-slim AS runtime

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
