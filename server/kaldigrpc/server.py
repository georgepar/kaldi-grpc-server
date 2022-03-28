import time
from concurrent import futures
from queue import Queue
from threading import Lock
import uuid
import pickle
import kaldigrpc.timeout_iterator as iterators
# import signal
import grpc
from kaldi.util.options import ParseOptions
from loguru import logger
import threading
import kaldigrpc.generated.asr_pb2 as msg
import kaldigrpc.generated.asr_pb2_grpc as rpc
from kaldigrpc.config import AsrConfig
from kaldigrpc.recognize import KaldiRecognizer, TimeoutException
import threading
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# def timeout_handler(signum, frame):  # Custom signal handler
#     raise TimeoutException


# Change the behavior of SIGALRM
# signal.signal(signal.SIGALRM, timeout_handler)


class ILSPASRService(rpc.ILSPASRServicer):
    def __init__(self, *args, **kwargs):
        if "asr_config" not in kwargs:
            raise ValueError(
                "Expecting an AsrConfig object as a kwarg. Try calling ILSPASRService(asr_config=my_asr_config)"
            )
        self.lock = threading.Lock()
        self.asr_config = kwargs.pop("asr_config")
        self.max_workers = kwargs.pop("max_workers")
        # self.available_workers = self.max_workers
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
            "/etc/certs/kaldi_key.pem",
            "SSL pem encoded private key. Required if --secure is passed, ignored otherwise",
        )
        po.register_str(
            "cert_chain",
            "/etc/certs/kaldi_chain.pem",
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

    def _make_asr(self):
        while self.available_workers == 0:
            time.sleep(500)
        with self.lock:
            self.available_workers -= 1
        asr = KaldiRecognizer(self.asr_config)
        return asr

    def Recognize(self, request, context):
        logger.log("INFO", "Server.Recognize: Batch recognition endpoint")
        metadata = dict(context.invocation_metadata())
        print(metadata)
        config = request.config
        print(config)

        logger.log("INFO", "Server.Recognize: Get next available worker from queue")
        # asr = self._make_asr()
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

        with open(f"/data/batch_{uuid.uuid1()}.wav", "wb") as fd:
            # pickle.dump((request, resp), fd)
            fd.write(request.audio)
        # with self.lock:
        #    self.available_workers += 1
        return resp

    def StreamingRecognize(self, request_iterator, context):

        logger.log("INFO", "Server.StreamingRecognize: Streaming recognition endpoint")
        wav_uuid = uuid.uuid1()
        req_iter = iterators.TimeoutIterator(request_iterator, timeout=3, sentinel=None)

        def stream():
            while True:
                try:
                    req = next(req_iter)
                    if req is None:
                        break
                    # signal.alarm(3
                    with open(f"/data/streaming_{wav_uuid}.wav", "ab") as fd:
                        fd.write(req.audio_content)
                except:
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

        stop_event = threading.Event()
        def on_rpc_done():
            # Regain servicer thread.
            stop_event.set()
            #asr.graceful_lattice_shutdown()
            logger.log("INFO", "Server.StreamingRecognize: STREAM ENDED Put worker back into the queue")
            self.recognizers.put(asr)  # Reinsert into the queue
        context.add_callback(on_rpc_done)
        # asr = self._make_asr()
        logger.log("INFO", "Server.StreamingRecognize: Iterating over requests")
        for idx, result in enumerate(asr.recognize_stream(stream())):
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

            # with open(f"/data/streaming_{wav_uuid}_resp_{idx}.p", "ab") as fd:
            #     pickle.dump(resp, fd)
            yield resp
        # with self.lock:
        #    self.available_workers += 1
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
        with open(args.cert_key, "rb") as fd:
            private_key = fd.read()

        with open(args.cert_chain, "rb") as fd:
            certificate_chain = fd.read()

        server_credentials = grpc.ssl_server_credentials(
            ((private_key, certificate_chain),)
        )

        server.add_secure_port(f"{args.host}:{args.port}", server_credentials)

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
