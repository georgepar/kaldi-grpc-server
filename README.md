# Kaldi gRPC Server

This is a modern alternative for deploying Speech Recognition models developed using Kaldi.

Features:

- Standardized API. We use a modified version of [Jarvis proto files](https://github.com/NVIDIA/speechsquad/blob/master/server/proto/jarvis_asr.proto#L53), which mimic the Google speech API.
 This allows for easy switching between Gloud speech recognizers and custom models developed with Kaldi
- Fully pythonic implementation. We utilize [pykaldi bindings](https://github.com/pykaldi/pykaldi) to interface with Kaldi programmatically. This allows for a clean, customizable and extendable implementation
- Fully bidirectional streaming using HTTP/2 (gRPC). Binary speech segments are streamed to the server and partial hypotheses are streamed back to the client
- Transcribe arbitrarily long speech
- DNN-HMM models supported out of the box
- Supports RNNLM lattice rescoring
- Clients for other languages can be easily generated using the proto files

## Getting started

### Kaldi model structure

We recommend the following structure for the deployed model

```
model
├── conf
│   ├── ivector_extractor.conf
│   ├── mfcc.conf
│   ├── online_cmvn.conf
│   ├── online.conf
│   └── splice.conf
├── final.mdl
├── global_cmvn.stats
├── HCLG.fst
├── ivector_extractor
│   ├── final.dubm
│   ├── final.ie
│   ├── final.mat
│   ├── global_cmvn.stats
│   ├── online_cmvn.conf
│   ├── online_cmvn_iextractor
│   └── splice_opts
└── words.txt
```

The key files / directories are:

- `conf`: Configuration files that are used to train the model
- `final.mdl`: The acoustic model
- `HCLG.fst`: The composed HCLG graph (output of mkgraph.sh)
- `global_cmvn.stats`: Mean and std used for CMVN normalization
- `words.txt`: Vocabulary file, mapping words to integers
- `ivector_extractor`: Model trained to extract ivector features (used for tdnn / chain models)


### Build Binary ASR recognizer (singularity)

We provide the option to build a (for all intents and purposes) binary file using the kaldi bindings through [singularity containers](https://sylabs.io/guides/3.0/user-guide/quick_start.html).
In short, singularity containers build a fakeroot filesystem into a single, executable file. For more info check the documentation.

Instructions:

- Install singularity on your machine. [Instructions here](https://sylabs.io/guides/3.0/user-guide/quick_start.html)
- Build the container
    ```
    make build-singularity kaldi_model=$MY_MODEL_DIR image_tag=asr
    # Do not include special characters like : in the image_tag argument because this will be the path to the container file
    ```
- Run the container with
    ```
    ./asr.sif --beam=11 --streaming --wav=$MYTEST.wav
    ```
- For more options run
    ```
    ./asr.sif --help
    ```

### Dockerized server deployment

Once you create this model structure, you can use the provided Dockerfile to build the server container. Run:

```
make build-server kaldi_model=$MY_MODEL_DIR image_tag=$CONTAINER_TAG
# example: make build-server kaldi_model=/models/kaldi/english_model image_tag=kaldigrpc:en-latest
```

And you can run the container

```
# Run your container for maximum 3 simultaneous clients on port 1234
make run-server image_tag=kaldigrpc:en-latest max_workers=3 server_port=1234

```

### Client usage

Install client library:

```
pip install kaldigrpc-client
```

Run client from command line:

```
kaldigrpc-transcribe --streaming --host localhost --port 50051 mytest.wav
```


For more infomation refer to `client/README.md`

### RNNLM Rescoring

TODO: Write documentation


### Roadmap

- Add support for mixed kaldi and pytorch acoustic / language models
- Add full support for pause detection (interim results)
- Add load balancer / benchmarks
- Streamined conversion scripts from exp folder to model tarball
- Support all Speech API configuration options
