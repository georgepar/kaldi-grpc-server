# Kaldi gRPC client

Python client library for Kaldi gRPC server. This client has similar - identical - semantics to the
[Google speech Python library](https://cloud.google.com/speech-to-text/docs/libraries#client-libraries-install-python).


## Installation

You can install from source

```bash
git clone https://github.com/georgepar/kaldi-grpc-server
cd client
pip install .
```

or from Pypi

```bash
pip install kaldigrpc-client
```

## Usage from command line

We assume you have a server running on port `50051`. See `kaldi-grpc-server` README for more
information.

```bash
kaldigrpc-transcribe --port 50051 $MY_WAV_FILE
```

For long files we recommend using the streaming client


```bash
kaldigrpc-transcribe --streaming --port 50051 $MY_WAV_FILE
```

## Programmatic usage

The following is a simple example for streaming recognition using the ILSPASRClient.
You can also refer to the code and the proto files for more configuration options and more outputs
(e.g. confidence, word start and end times etc.)

**Warning**: Some configuration options are included for compatibility / easy swapping with the Google Speech
client library but are not yet fully implemented. Please refer to the code for more details.

```python
cli = ILSPASRClient(host="localhost", port=50051)

chunks = ...  # list of audio chunks (bytes)

for partial_result in cli.streaming_recognize(chunks):
    # Print best path partial transcription
    print(partial_result.results[0].alternatives[0].transcript)
```
