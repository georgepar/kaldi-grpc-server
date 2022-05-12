from __future__ import print_function
from kaldi.util.table import MatrixWriter, SequentialMatrixReader
from kaldi.util.options import ParseOptions
from kaldigrpc.config import DiarizerConfig, SadConfig
from kaldigrpc.segment import KaldiSegmentor
import sys
import os
import subprocess
from os.path import dirname, join
sys.path.insert(0, join(dirname(__file__), '..'))
from scripts import xvectors, plda, cluster, split_wav


class KaldiDiarizer:

    def __init__(self, cfg: DiarizerConfig):
        self.config = cfg

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("wav", "", "Path to wav file for decoding")
        po.register_bool("streaming", False, "Use streaming online segmentor")

    def diarize(self, wav_path):
        # Create segments file with the help of SAD component
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
        os.system("cp /kaldigrpc/SAD_model/data/segments /kaldigrpc/diarizer_model/data/")

        # Create wav.scp
        with open("/kaldigrpc/diarizer_model/data/wav.scp", "w") as f:
            f.write("utt1 " + wav_path)

        ### Segments extraction from wav
        split_wav.extract_segments("scp:/kaldigrpc/diarizer_model/data/wav.scp", "/kaldigrpc/diarizer_model/data/segments",
                                  "ark,scp:/kaldigrpc/diarizer_model/data/wav2.ark,/kaldigrpc/diarizer_model/data/wav2.scp", self.config.min_segment)
        
        ### Create additional necessary files
        # create utt2spk
        os.system(
            "awk -v var="+str(self.config.min_segment)+" '{if ($4 - $3 >= var) {print $1, $2}}' /kaldigrpc/diarizer_model/data/segments > /kaldigrpc/diarizer_model/data/utt2spk")
        # create spk2utt
        with open("/kaldigrpc/diarizer_model/data/spk2utt", 'a+') as file:
            subprocess.Popen(["../scripts/utt2spk_to_spk2utt.pl",
                             "/kaldigrpc/diarizer_model/data/utt2spk"], stdout=file)
        # remove minimum segments from segments file
        os.system(
            "awk -v var="+str(self.config.min_segment)+" '{if ($4 - $3 >= var) {print $1, $2, $3, $4}}' /kaldigrpc/diarizer_model/data/segments > /kaldigrpc/diarizer_model/data/filtered_segments")

        ### Feature Computation
        print("\nComputing mfcc features...")
        feats_rspecifier = (
            "ark:compute-mfcc-feats --config=/kaldigrpc/model/conf/mfcc.conf scp:/kaldigrpc/diarizer_model/data/wav2.scp ark:- |"
        )

        with MatrixWriter("ark,scp:../diarizer_model/data/feats.ark,../diarizer_model/data/feats.scp") as writer:
            for key, feats in SequentialMatrixReader(feats_rspecifier):
                writer[key] = feats
        print("Operation complete!")

        ### X-vector Extraction
        xvectors.extract_vectors(self.config, apply_cmn=False, window=1.5)
        ### PLDA Scoring
        plda.score_plda(self.config, target_energy=0.5) 
        ### Clustering
        cluster.apply_clustering(self.config, rttm_channel=1)

def transcribe_wav():
    po = ParseOptions(
        """Diarization using Kaldi.
        Usage:
            python diarize.py --wav=/path/to/test.wav
        """
    )

    KaldiDiarizer.register(po)
    config, args = DiarizerConfig.parse_options(po)
    diarizer = KaldiDiarizer(config)
    diarizer.diarize(args.wav)


if __name__ == "__main__":
    transcribe_wav()
