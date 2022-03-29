import argparse
from typing import Iterator, Optional

import grpc
from google.protobuf.json_format import MessageToDict

from .generated import asr_pb2 as msg
from .generated import asr_pb2_grpc as rpc


class ILSPASRClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        # TODO: Unused args for now. Adding for compatibility with google speech client
        secure: bool = False,
        chunk_size: int = 200,
        streaming_output: bool = False,
        encoding: str = "LINEAR16",
        sample_rate_hertz: int = 16000,
        language_code: str = "el-GR",
        max_alternatives: int = 1,
        audio_channel_count: int = 1,
        enable_automatic_punctuation: bool = False,
        enable_word_time_offsets: bool = False,
        enable_separate_recognition_per_channel: bool = False,
        model: str = "generic",
        interim_results: bool = True,
    ) -> None:

        encoding = msg.AudioEncoding.LINEAR_PCM  # encoding=LINEAR16
        self.recognition_config = msg.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate_hertz,
            language_code=language_code,
            max_alternatives=max_alternatives,
            audio_channel_count=audio_channel_count,
            enable_automatic_punctuation=enable_automatic_punctuation,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_separate_recognition_per_channel=enable_separate_recognition_per_channel,
            model=model,
        )

        self.streaming_config = msg.StreamingRecognitionConfig(
            config=self.recognition_config, interim_results=interim_results
        )
        self.streaming_output = streaming_output
        self.chunk_size = chunk_size

        if not secure:
            channel = grpc.insecure_channel(
                f"{host}:{port}",
                options=(
                    ("grpc.enable_http_proxy", 0),
                    ("grpc.max_message_length", 10366164 * 2),
                ),
            )
        else:
            raise ValueError("SSL not yet supported")
            # channel = grpc.secure_channel(...)

        self.connection = rpc.ILSPASRStub(channel)

    def recognize(
        self, wavbytes: bytes, config: Optional[msg.RecognitionConfig] = None
    ):
        req = msg.RecognizeRequest(
            config=config or self.recognition_config, audio=wavbytes
        )

        return self.connection.Recognize(req)

    def streaming_recognize_requests(
        self, requests: Iterator[msg.StreamingRecognizeRequest]
    ):
        for res in self.connection.StreamingRecognize(requests()):
            yield res

    def streaming_recognize(
        self,
        chunks: Iterator[bytes],
        config: Optional[msg.StreamingRecognitionConfig] = None,
    ):
        def requests_iter():
            req = msg.StreamingRecognizeRequest(
                streaming_config=config or self.streaming_config
            )

            for chunk in chunks:
                req = msg.StreamingRecognizeRequest(audio_content=chunk)
                yield req

        for res in self.connection.StreamingRecognize(requests_iter()):
            yield res

    def recognize_wav(self, wav_file: str):
        from pydub import AudioSegment

        res = self.recognize(AudioSegment.from_file(wav_file).raw_data)

        print(res.results[0].alternatives[0].transcript)
        return res

    def recognize_streaming_wav(self, wav_file: str):
        from pydub import AudioSegment
        from pydub.utils import make_chunks

        snd = AudioSegment.from_file(wav_file)

        def wav_iter():
            for chunk in make_chunks(snd, self.chunk_size):
                yield chunk.raw_data

        recognizer_outputs = self.streaming_recognize(wav_iter())

        response = None

        while True:
            try:
                response = next(recognizer_outputs)

                if self.streaming_output:
                    print(response.results[0].alternatives[0].transcript, end="\r")

            except StopIteration:
                break

        if response is not None:
            print(response.results[0].alternatives[0].transcript)

        return response

    def write_srt(self, response, out_srt, max_chars=40):
        import srt

        subs = []
        for result in response.results:
            alternative = result.alternatives[0]
            firstword = True
            charcount = 0
            idx = len(subs) + 1
            content = ""

            for w in alternative.words:
                if firstword:
                    # first word in sentence, record start time
                    start = w.start_time.ToTimedelta()
                if (
                    "." in w.word
                    or "!" in w.word
                    or "?" in w.word
                    or charcount > max_chars
                    or ("," in w.word and not firstword)
                ):
                    # break sentence at: . ! ? or line length exceeded
                    # also break if , and not first word
                    subs.append(
                        srt.Subtitle(
                            index=idx,
                            start=start,
                            end=w.end_time.ToTimedelta(),
                            content=srt.make_legal_content(content),
                        )
                    )
                    firstword = True
                    idx += 1
                    content = ""
                    charcount = 0
                else:
                    firstword = False

        with open(out_srt, "w") as f:
            f.writelines(srt.compose(subs))

    def write_alignments(self, result):
        result_dict = MessageToDict(result)

        word_list = result_dict["results"][0]["alternatives"][0]["words"]

        for d in word_list:
            d["speaker"] = "speakerID"
            d["speaker_name"] = "Anonymous"

            if "startTime" not in d:
                d["startTime"] = -1

            if "endTime" not in d:
                d["endTime"] = -1

        with open(out_csv, "w") as alf:
            for elem in word_list:
                ln = f"{elem['speaker']}\t{elem['speaker_name']}\t{elem['startTime']}\t{elem['endTime']}\t{elem['word']}\n"
                alf.write(ln)

    @classmethod
    def register(cls, parser: argparse.ArgumentParser):
        parser.add_argument("wav", type=str, help="Wav file to be transcribed")
        parser.add_argument(
            "--streaming", action="store_true", help="Use streaming transcription"
        )
        parser.add_argument(
            "--streaming-output", action="store_true", help="Print streaming output"
        )
        parser.add_argument(
            "--alignment-output",
            type=str,
            default=None,
            help="Path to write the alignment csv",
        )
        parser.add_argument(
            "--srt-output",
            type=str,
            default=None,
            help="Path to write the srt",
        )
        parser.add_argument("--host", type=str, default="localhost", help="Server host")
        parser.add_argument("--port", type=int, default=50051, help="Server port")
        parser.add_argument(
            "--chunk-size", type=int, default=200, help="Chunk size for wav recognition"
        )
        # TODO: Add meaningful support for other configurations

        return parser


def transcribe_wav():
    parser = argparse.ArgumentParser(
        """Kaldi GRPC client
        Usage:
            python client.py --streaming --wav test.wav
        """
    )
    parser = ILSPASRClient.register(parser)
    args = parser.parse_args()
    cli = ILSPASRClient(
        host=args.host,
        port=args.port,
        streaming_output=args.streaming_output,
    )

    if args.streaming:
        result = cli.recognize_streaming_wav(args.wav)
    else:
        result = cli.recognize_wav(args.wav)

    if args.alignment_output is not None:
        cli.write_alignments(result, args.alignment_output)

    if args.srt_output is not None:
        cli.write_srt(result, args.srt_output)


if __name__ == "__main__":
    transcribe_wav()
