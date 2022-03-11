PIP=pip
PYTHON=python

max_workers=2
server_port=50051

all: depend generate-stubs

depend:
	$(PIP) install poetry grpcio-tools 2to3

generate-stubs:
	$(PYTHON) -m grpc_tools.protoc -I./protos --python_out=server/kaldigrpc/generated --grpc_python_out=server/kaldigrpc/generated protos/asr.proto
	$(PYTHON) -m grpc_tools.protoc -I./protos --python_out=client/kaldigrpc_client/generated --grpc_python_out=client/kaldigrpc_client/generated protos/asr.proto
	2to3 server/kaldigrpc/generated -w -n
	2to3 client/kaldigrpc_client/generated -w -n

build-server:
	# Due to the overhead of installing kaldi / pykaldi, containerized build is the most sane
	# approach
	cd server && ./build-docker.sh $(kaldi_model) $(image_tag)

build-singularity:
	# Due to the overhead of installing kaldi / pykaldi, containerized build is the most sane
	# approach
	cd server && ./build-singularity.sh $(kaldi_model) $(image_tag) && cp $(image_tag).sif ../containers/

build-flex-singularity:
	# Build singularity container without preinstalled model
	cd server && singularity build --notest --fakeroot asr.sif asr-flex-singularity.def && cp asr.sif ../containers/


run-server:
	docker run -p $(server_port):$(server_port) -ti $(image_tag) --port=$(server_port) --max-workers=$(max_workers)

build-client:
	cd client && poetry install && poetry build

publish-client:
	cd client && poetry install && poetry build && poetry publish

clean:
	rm -rf server/model/*
