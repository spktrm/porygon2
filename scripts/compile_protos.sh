#!/bin/bash

. env/bin/activate
echo "Generating messages and enums protos..."
python proto/scripts/make_enums.py

protolint --fix proto/
if [ $? -ne 0 ]; then
  echo "Protolint has fixed some things!"
fi

# Navigate to the server directory and run the TypeScript compilation script
echo "Compiling Protobuf files for TypeScript..."
cd service

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

if [ $? -ne 0 ]; then
  echo "TypeScript Protobuf compilation failed!"
  exit 1
fi

cd ../

# Run the Python Protobuf compilation command
echo "Compiling Protobuf files for Python..."

rm -rf rl/environment/protos/
mkdir rl/environment/protos/

python -m grpc_tools.protoc -Iproto --python_out=rl/environment/protos/ --pyi_out=rl/environment/protos/ --grpc_python_out=rl/environment/protos/ proto/*.proto

if [ $? -ne 0 ]; then
  echo "Python Protobuf compilation failed!"
  exit 1
fi

fix-protobuf-imports rl/environment/protos/

echo "Protobuf compilation completed successfully!"