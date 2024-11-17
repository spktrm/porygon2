rm -rf protos/
mkdir protos/

python -m grpc_tools.protoc -I../proto --python_out=protos/ --pyi_out=protos/ --grpc_python_out=protos/ ../proto/*.proto
