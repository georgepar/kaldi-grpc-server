from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import loguru

from kaldi.asr import (LatticeRnnlmPrunedRescorer,
                       NnetLatticeFasterOnlineRecognizer)
from kaldi.fstext import SymbolTable
from kaldi.lat.sausages import MinimumBayesRisk
from kaldi.online2 import (OnlineIvectorExtractorAdaptationState,
                           OnlineNnetFeaturePipeline, OnlineSilenceWeighting)
from kaldi.rnnlm import RnnlmComputeStateComputationOptions
from kaldi.util.options import ParseOptions
from kaldigrpc.config import AsrConfig
from kaldigrpc.util import timefn, timegen
from kaldigrpc.wav import bytes2vector


@dataclass
class WordInfo:
    start_time: float
    end_time: float
    word: str


@dataclass
class Transcription:
    transcript: str
    confidence: float
    words: List[WordInfo]
    is_final: bool = True


class KaldiRecognizer:
    def __init__(self, cfg: AsrConfig):
        self.config = cfg
        self.asr = NnetLatticeFasterOnlineRecognizer.from_files(
            self.config.model.finalmdl,
            self.config.model.hclgfst,
            self.config.model.wordstxt,
            decoder_opts=self.config.decoder,
            decodable_opts=self.config.decodable,
            endpoint_opts=self.config.endpoint,
        )

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("wav", "test.wav", "Path to wav file for decoding")
        po.register_bool("streaming", False, "Use streaming online recognizer")

    def _load_rnnlm(self) -> Optional[LatticeRnnlmPrunedRescorer]:
        if not self.config.rnnlm.use_rnnlm:
            return None

        word_embedding_rxfilename = "rnnlm-get-word-embedding %s %s - |" % (
            self.config.rnnlm.word_feats,
            self.config.rnnlm.feat_embedding,
        )

        # Instantiate the rescorer
        symbols = SymbolTable.read_text(self.config.rnnlm.vocab)
        opts = RnnlmComputeStateComputationOptions()
        opts.bos_index = symbols.find_index(self.config.rnnlm.bos)
        opts.eos_index = symbols.find_index(self.config.rnnlm.eos)
        opts.brk_index = symbols.find_index(self.config.rnnlm.brk)
        rescorer = LatticeRnnlmPrunedRescorer.from_files(
            self.config.rnnlm.old_lm,
            word_embedding_rxfilename,
            self.config.rnnlm.rnnlm_model,
            opts=opts,
            acoustic_scale=1.0,
            lm_scale=0.2,
        )

        return rescorer

    def _words_from_indices(
        self, words: List[int], times: Optional[List[Tuple[int, int]]] = None
    ) -> List[WordInfo]:
        def get_time(i, times, st=True):
            if times is None:
                return 0
            else:
                idx = 0 if st else 1

                return times[i][idx]

        return [
            WordInfo(
                start_time=get_time(i, times, st=True),
                end_time=get_time(i, times, st=False),
                word=self.asr.symbols.find_symbol(w),
            )

            for i, w in enumerate(words)
        ]

    def _mbr_transcription(self, mbr: MinimumBayesRisk) -> Transcription:
        confidences = mbr.get_one_best_confidences()
        confidence = sum(confidences) / len(confidences)
        mbr_trans = " ".join(
            [self.asr.symbols.find_symbol(i) for i in mbr.get_one_best()]
        )

        transcript = Transcription(
            mbr_trans,
            confidence,
            self._words_from_indices(
                mbr.get_one_best(), times=mbr.get_one_best_times()
            ),
            is_final=True,
        )

        return transcript

    @timefn(method=True)
    def recognize(self, wavbytes: bytes) -> Transcription:
        # Decode (whole utterance)
        wav = bytes2vector(wavbytes, self.config.audio)
        feat_pipeline = OnlineNnetFeaturePipeline(self.config.feat)
        self.asr.set_input_pipeline(feat_pipeline)
        rescorer = self._load_rnnlm()
        feat_pipeline.accept_waveform(self.config.audio.frame_rate, wav)
        feat_pipeline.input_finished()
        out = self.asr.decode()

        rescored_lat = out["lattice"]

        if rescorer is not None:
            rescored_lat = rescorer.rescore(out["lattice"])

        mbr = MinimumBayesRisk(rescored_lat)
        transcript = self._mbr_transcription(mbr)

        return transcript

    @timegen(method=True)
    def recognize_stream(self, chunks: Iterator[bytes]) -> Iterator[Transcription]:
        # Decode (chunked + partial output)
        adaptation_state = OnlineIvectorExtractorAdaptationState.from_info(
            self.config.feat.ivector_extractor_info
        )
        rescorer = self._load_rnnlm()

        feat_pipeline = OnlineNnetFeaturePipeline(self.config.feat)
        feat_pipeline.set_adaptation_state(adaptation_state)

        self.asr.set_input_pipeline(feat_pipeline)
        self.asr.init_decoding()

        silence_weighting = OnlineSilenceWeighting(
            self.asr.transition_model,
            self.config.feat.silence_weighting_config,
            self.config.decodable.frame_subsampling_factor,
        )

        last_chunk = False
        prev_num_frames_decoded = 0

        while True:
            try:
                chunk_bytes = next(chunks)
                chunk = bytes2vector(chunk_bytes, self.config.audio)
            except StopIteration:
                last_chunk = True

            if last_chunk:
                feat_pipeline.input_finished()

                self.asr.finalize_decoding()
                out = self.asr.get_output()

                rescored_lat = out["lattice"]

                if rescorer is not None:
                    rescored_lat = rescorer.rescore(out["lattice"])

                mbr = MinimumBayesRisk(rescored_lat)
                transcript = self._mbr_transcription(mbr)
                yield transcript

                break

            feat_pipeline.accept_waveform(self.config.audio.frame_rate, chunk)

            if silence_weighting.active():
                silence_weighting.compute_current_traceback(self.asr.decoder)
                feat_pipeline.ivector_feature().update_frame_weights(
                    silence_weighting.get_delta_weights(
                        feat_pipeline.num_frames_ready()
                    )
                )

            self.asr.advance_decoding()
            num_frames_decoded = self.asr.decoder.num_frames_decoded()

            if num_frames_decoded > prev_num_frames_decoded:
                prev_num_frames_decoded = num_frames_decoded
                out = self.asr.get_partial_output()

                transcript = Transcription(
                    out["text"],
                    out["likelihood"],
                    self._words_from_indices(out["words"]),
                    is_final=False,
                )
                yield transcript

    def recognize_wav(self, wav_file):
        from pydub import AudioSegment

        wb = AudioSegment.from_file(wav_file).raw_data

        result = self.recognize(wb)
        print(result.transcript)

    def recognize_streaming_wav(self, wav_file):
        from pydub import AudioSegment
        from pydub.utils import make_chunks

        snd = AudioSegment.from_file(wav_file)

        def wav_stream():
            for chunk in make_chunks(snd, 100):
                yield chunk.raw_data

        for result in self.recognize_stream(wav_stream()):
            print(result.transcript)


def transcribe_wav():
    po = ParseOptions(
        """Online decoding using Kaldi.
        Usage:
            python recognize.py --model-dir=/model/ --wav=/path/to/test.wav
        """
    )
    KaldiRecognizer.register(po)
    config, args = AsrConfig.parse_options(po=po)
    asr = KaldiRecognizer(config)

    if args.streaming:
        asr.recognize_streaming_wav(args.wav)
    else:
        asr.recognize_wav(args.wav)


if __name__ == "__main__":
    transcribe_wav()
