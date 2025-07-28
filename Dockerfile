FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, Node.js, tmux
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl tmux git && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Install TypeScript and ts-node globally
RUN npm install -g typescript ts-node
RUN npm install -g @types/node

# Install protobuf compiler (protoc)
RUN apt-get update && apt-get install -y \
    protobuf-compiler

# Install protolint (from GitHub release)
RUN curl -L https://github.com/yoheimuta/protolint/releases/download/v0.55.6/protolint_0.55.6_linux_amd64.tar.gz \
    | tar -xz && \
    mv protolint /usr/local/bin/protolint && \
    chmod +x /usr/local/bin/protolint

# Set working directory
WORKDIR /app

# Copy package.json for all apps
COPY app*/package*.json ./

# Install for each
RUN for dir in app*/; do \
    cd "$dir" && npm install && cd ..; \
    done

# Copy the rest after install
COPY . .

# Pull the latest changes for the Pokemon Showdown submodule
# TODO: Might be unnecessary...
# RUN cd data && \
#     git submodule update --init --recursive && \
#     cd ps && git checkout main && git pull origin main

ENV PYTHONPATH="/app:${PYTHONPATH}"

RUN cd /app && pip install virtualenv && virtualenv env && \
    . env/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install -U "jax[cuda12]"

# Generate indices from pokemon showdown
RUN make data

RUN python protos/scripts/make_enums.py

RUN make protos/

CMD ["/start.sh"]
