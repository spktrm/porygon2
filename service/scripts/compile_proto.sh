#!/bin/bash

rm -rf protos/
mkdir protos/

PROTO_DIR=../proto
OUT_DIR=../service/protos

mkdir -p $OUT_DIR

npx protoc \
    --plugin=protoc-gen-ts=./node_modules/.bin/protoc-gen-ts \
    --js_out=import_style=commonjs,binary:$OUT_DIR \
    --ts_out=service=grpc-web:$OUT_DIR \
    -I $PROTO_DIR $PROTO_DIR/*.proto
