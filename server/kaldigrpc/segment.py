#!/usr/bin/env python

from __future__ import print_function

from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialMatrixReader
from kaldigrpc.config import SadConfig

class KaldiSegmentor:
    def __init__(self, cfg: SadConfig, args):
        self.config = cfg
        self.wav = args.wav

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("wav", "", "Path to wav file for decoding")
        po.register_bool("streaming", False, "Use streaming online segmentor")

    def segment(self):

        with open(self.config.model.wavscp, "w+") as f:
            f.write("utt1 " + self.wav)

        with SequentialMatrixReader(self.config.feats_rspec) as f, open(
            "/kaldigrpc/SAD_model/data/segments", "w+"
        ) as s:
            for key, feats in f:
                out = self.config.sad.segment(feats)
                segments, stats = self.config.seg.process(out["alignment"])
                self.config.seg.write(key, segments, s)
                # print("segments:", segments, flush=True)
                # print("stats:", stats, flush=True)
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
