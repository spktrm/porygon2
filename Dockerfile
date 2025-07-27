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

# Copy everything except ignored content
COPY . .

# Install Node.js deps only in subdirs that have package.json
RUN for dir in */; do \
    if [ -f "$dir/package.json" ]; then \
    cd "$dir" && npm install && cd ..; \
    fi; \
    done

# Pull the latest changes for the Pokemon Showdown submodule
# TODO: Might be unnecessary...
# RUN cd data && \
#     git submodule update --init --recursive && \
#     cd ps && git checkout main && git pull origin main

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install -U "jax[cuda12]"

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Generate indices from pokemon showdown
RUN make data

RUN python3 protos/scripts/make_enums.py

RUN make protos/

# Create tmux runner for two apps (adjust as needed)
RUN echo '#!/bin/bash\n\
    tmux new-session -d -s multi "cd /service && npm run start"\n\
    tmux split-window -v -t multi "python rl/main.py"\n\
    tmux attach -t multi' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
