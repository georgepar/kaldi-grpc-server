import time
from concurrent import futures
from queue import Queue
from threading import Lock
import kaldigrpc.timeout_iterator as iterators

import grpc
from kaldi.util.options import ParseOptions
from loguru import logger

import kaldigrpc.generated.asr_pb2 as msg
import kaldigrpc.generated.asr_pb2_grpc as rpc
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
            logger.log("INFO", "Final result. Constructing alternative")
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
            logger.log("INFO", "Not final yet. Partial result")
            alt = msg.SpeechRecognitionAlternative(transcript=result.transcript)

        return [alt]

    def Recognize(self, request, context):
        logger.log("INFO", "Server.Recognize: Batch recognition endpoint")
        metadata = dict(context.invocation_metadata())
        print(metadata)
        config = request.config
        print(config)

        logger.log("INFO", "Server.Recognize: Get next available worker from queue")
        asr = self.recognizers.get(
            block=True, timeout=None
        )  # Block until a recognizer becomes available
        logger.log("INFO", "Server.Recognize: Recognition started")
        result = asr.recognize(request.audio)
        logger.log("INFO", "Server.Recognize: Recognition finished")

        res = msg.SpeechRecognitionResult(
            alternatives=self._construct_alternatives(result),
            channel_tag=0,  # FIXME: Support single channel for now.
            audio_processed=0,  # FIXME: Do not return audio duration for now
        )
        resp = msg.RecognizeResponse(results=[res])
        logger.log("INFO", "Server.Recognize: Put worker back into the queue")
        self.recognizers.put(asr)  # Reinsert into the queue
        logger.log("INFO", "Server.Recognize: Return response to client")
        return resp

    def StreamingRecognize(self, request_iterator, context):
        logger.log("INFO", "Server.StreamingRecognize: Streaming recognition endpoint")
        req_iter = iterators.TimeoutIterator(request_iterator, timeout=3)

        def stream():
            while True:
                try:
                    req = next(req_iter)
                    if req is req_iter.get_sentinel():
                        logger.log(
                            "INFO", "Timeout has been reached. Closing stream..."
                        )
                        context.cancel()
                        break
                except StopIteration:
                    break

                if not req.audio_content:
                    config = req.config
                    print(config)
                else:
                    yield req.audio_content

        logger.log(
            "INFO", "Server.StreamingRecognize: Get next available worker from queue"
        )
        asr = self.recognizers.get(
            block=True, timeout=None
        )  # Block until a recognizer becomes available

        # clear the stop_event here before starting using the ASR
        asr.stop_event.clear()

        def on_rpc_done():
            # regain servicer thread
            asr.stop_event.set()

        # add on_rpc_done as callback function when a RPC communication is terminated
        context.add_callback(on_rpc_done)

        logger.log("INFO", "Server.StreamingRecognize: Iterating over requests")
        for result in asr.recognize_stream(stream()):

            # block the return of a response to a closed channel
            # and continue to the next iteration so as to
            # force-follow the "last_chunk" path (to end ASR decoding gracefully)
            if asr.stop_event.is_set():
                continue

            res = msg.StreamingRecognitionResult(
                alternatives=self._construct_alternatives(result),
                is_final=result.is_final,
                channel_tag=0,
                audio_processed=0,
            )
            logger.log(
                "INFO",
                "Server.StreamingRecognize: Processed chunk. Returning partial result to client",
            )

            resp = msg.StreamingRecognizeResponse(results=[res])

            yield resp

        logger.log("INFO", "Server.StreamingRecognize: Put worker back into the queue")
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
