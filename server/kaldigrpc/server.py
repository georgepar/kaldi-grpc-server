import time
from concurrent import futures
from queue import Queue
from threading import Lock

import grpc
from loguru import logger

import kaldigrpc.generated.asr_pb2 as msg
import kaldigrpc.generated.asr_pb2_grpc as rpc
from kaldi.util.options import ParseOptions
from kaldigrpc.config import AsrConfig
from kaldigrpc.recognize import KaldiRecognizer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ILSPASRService(rpc.ILSPASRServicer):
    def __init__(self, *args, **kwargs):
        if "asr_config" not in kwargs:
            raise ValueError(
                "Expecting an AsrConfig object as a kwarg. Try calling ILSPASRService(asr_config=my_asr_config)"
            )
        self.asr_config = kwargs.pop("asr_config")
        self.max_workers = kwargs.pop("max_workers")
        self.recognizers = Queue(maxsize=self.max_workers)

        for _ in range(self.max_workers):
            self.recognizers.put(KaldiRecognizer(self.asr_config))
        super(ILSPASRService, self).__init__(*args, **kwargs)

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("host", "0.0.0.0", "Service host IP")
        po.register_str("port", "50051", "Service port")
        po.register_int(
            "max_workers", 10, "Maximum number of threads to be used for serving"
        )
        po.register_bool(
            "secure",
            False,
            "Use ssl/tls authentication. Requires --cert-key and --cert-chain to be passed",
        )
        po.register_str(
            "cert_key",
            "key.pem",
            "SSL pem encoded private key. Required if --secure is passed, ignored otherwise",
        )
        po.register_str(
            "cert_chain",
            "chain.pem",
            "SSL pem encoded certificate chain. Required if --secure is passed, ignored otherwise",
        )

    def _construct_alternatives(self, result):
        if result.is_final:
            alt = msg.SpeechRecognitionAlternative(
                transcript=result.transcript,
                confidence=result.confidence,
                words=[
                    msg.WordInfo(
                        start_time=w.start_time, end_time=w.end_time, word=w.word
                    )

                    for w in result.words
                ],
            )
        else:
            alt = msg.SpeechRecognitionAlternative(transcript=result.transcript)

        return [alt]

    def Recognize(self, request, context):
        metadata = dict(context.invocation_metadata())
        print(metadata)
        config = request.config
        print(config)
        asr = self.recognizers.round_robin()  # KaldiRecognizer(self.asr_config)
        result = asr.recognize(request.audio)

        res = msg.SpeechRecognitionResult(
            alternatives=self._construct_alternatives(result),
            channel_tag=0,  # FIXME: Support single channel for now.
            audio_processed=0,  # FIXME: Do not return audio duration for now
        )
        resp = msg.RecognizeResponse(results=[res])

        return resp

    def StreamingRecognize(self, request_iterator, context):
        def stream():
            while True:
                try:
                    req = next(request_iterator)
                except:
                    break

                if not req.audio_content:
                    config = req.config
                    print(config)
                else:
                    yield req.audio_content

        asr = self.recognizers.get(
            block=True, timeout=None
        )  # Block until a recognizer becomes available

        for result in asr.recognize_stream(stream()):
            res = msg.StreamingRecognitionResult(
                alternatives=self._construct_alternatives(result),
                is_final=result.is_final,
                channel_tag=0,
                audio_processed=0,
            )

            resp = msg.StreamingRecognizeResponse(results=[res])

            yield resp

        self.recognizers.put(asr)  # Reinsert into the queue


def serve():
    po = ParseOptions(
        """Kaldi GRPC server
        Usage:
            python server.py --model-dir=/model/ --host=0.0.0.0 --port=50051 --max_workers=10
        """
    )
    ILSPASRService.register(po)
    config, args = AsrConfig.parse_options(po=po)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    rpc.add_ILSPASRServicer_to_server(
        ILSPASRService(asr_config=config, max_workers=args.max_workers), server
    )

    if not args.secure:
        server.add_insecure_port(f"{args.host}:{args.port}")
    else:
        with open(args.cert_key, "r") as fd:
            private_key = fd.read()

        with open(args.cert_chain, "r") as fd:
            certificate_chain = fd.read()

        server_credentials = grpc.ssl_server_credentials(
            ((private_key, certificate_chain),)
        )

        server.add_secure_port("{args.host}:{args.port}", server_credentials)

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
