FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: Python, Node.js, tmux, git, protobuf compiler
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        tmux \
        git \
        protobuf-compiler && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TypeScript and ts-node globally
RUN npm install -g typescript ts-node @types/node

# Install protolint (from GitHub release)
RUN curl -L https://github.com/yoheimuta/protolint/releases/download/v0.55.6/protolint_0.55.6_linux_amd64.tar.gz \
    | tar -xz -C /tmp && \
    mv /tmp/protolint /usr/local/bin/protolint && \
    chmod +x /usr/local/bin/protolint

WORKDIR /app

# Install Node.js dependencies for the WebSocket service
COPY service/package*.json service/
RUN cd service && npm install

# Install Node.js dependencies for data generation
COPY data/package*.json data/
RUN cd data && npm install

# Copy the rest of the application
COPY . .

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Create a virtual environment and install all Python dependencies
# (requirements.txt already includes JAX with CUDA 13 support)
RUN python3 -m venv env && \
    . env/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Generate indices from Pokemon Showdown data
RUN . env/bin/activate && make datas

# Compile protocol buffer definitions for Python and TypeScript
RUN . env/bin/activate && make protos

CMD ["/app/start.sh"]
