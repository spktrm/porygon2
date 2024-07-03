#!/bin/bash

# Navigate to the server directory and run the TypeScript compilation script
echo "Compiling Protobuf files for TypeScript..."
cd server
./scripts/compile_proto.sh
if [ $? -ne 0 ]; then
  echo "TypeScript Protobuf compilation failed!"
  exit 1
fi
cd ../rlenv

# Run the Python Protobuf compilation command
echo "Compiling Protobuf files for Python..."
./scripts/compile_proto.sh
if [ $? -ne 0 ]; then
  echo "Python Protobuf compilation failed!"
  exit 1
fi

echo "Protobuf compilation completed successfully!"
