python -m grpc_tools.protoc -I./protos --python_out=kaldigrpc/generated --grpc_python_out=kaldigrpc/generated protos/asr.proto
2to3 kaldigrpc/generated -w -n
