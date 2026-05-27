FROM python:3.11.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential gfortran git tini \
    liblapack3 liblapack-dev libblas3 libblas-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY freegsnke/ ./freegsnke/

# Export deps from lockfile (excludes the project itself).
# morph_gs code will be mounted from /DATALAKE at runtime via PYTHONPATH.
RUN uv export --format requirements-txt --no-emit-project \
        -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt && \
    uv pip install --system --no-deps ./freegsnke/

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
