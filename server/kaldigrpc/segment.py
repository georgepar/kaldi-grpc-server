#!/usr/bin/env python

from __future__ import print_function
import random

from kaldi.util.table import SequentialMatrixReader
from kaldigrpc.config import SadConfig

from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineIvectorExtractorAdaptationState,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo,
                           OnlineNnetFeaturePipeline,
                           OnlineSilenceWeighting)

chunk_size = 300000


class KaldiSegmentor:

    def __init__(self, cfg: SadConfig, args):
        self.config = cfg
        self.wav = args.wav

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("wav", "", "Path to wav file for decoding")
        po.register_bool("streaming", False, "Use streaming online segmentor")

    def segment(self):

        f = open(self.config.model.wavscp, "w")
        f.write(str(random.randrange(100000, 1000000)) + " " + self.wav + "\n")
        f.close()

        with SequentialMatrixReader(self.config.feats_rspec) as f, open("segments", "w") as s:
            for key, feats in f:
                out = self.config.sad.segment(feats)
                segments, stats = self.config.seg.process(out["alignment"])
                self.config.seg.write(key, segments, s)
                print("segments:", segments, flush=True)
                print("stats:", stats, flush=True)
        print("global stats:", self.config.seg.stats, flush=True)

def transcribe_wav():
    po = ParseOptions(
        """SAD using Kaldi.
        Usage:
            python segment.py --model-dir=/model/ --wav=/path/to/test.wav
        """
    )

    KaldiSegmentor.register(po)
    config, args = SadConfig.parse_options(po)
    sad = KaldiSegmentor(config, args)
    sad.segment()


if __name__ == "__main__":
    transcribe_wav()
