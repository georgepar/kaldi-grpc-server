from kaldi.util.table import SequentialMatrixReader, VectorWriter
from kaldi.matrix import Vector
from kaldi.util.table import SequentialMatrixReader
from kaldi.ivector import compute_vad_energy, VadEnergyOptions
from kaldi.util.options import ParseOptions


def compute_vad(feat_rspecifier, vad_wspecifier):
    try:
        omit_unvoiced_utts = False

        po = ParseOptions("")
        opts = VadEnergyOptions()
        opts.register(po)
        po.read_config_file("/kaldigrpc/diarizer_model/src/vad.conf")

        feat_reader = SequentialMatrixReader(feat_rspecifier)
        vad_writer = VectorWriter(vad_wspecifier)

        num_done = 0
        num_err = 0
        num_unvoiced = 0
        tot_length = 0.0
        tot_decision = 0.0

        for utt, feat in feat_reader:
            if (feat.num_rows == 0):
                print("Warning: Empty feature matrix for utterance " + str(utt))
                num_err += 1
                continue

            vad_result = Vector(feat.num_rows)
            vad_result = compute_vad_energy(opts, feat)

            sum = vad_result.sum()

            if (sum == 0.0):
                print("Warning: No frames were judged voiced for utterance " + str(utt))
                num_unvoiced += 1
            else:
                num_done += 1

            tot_decision += vad_result.sum()
            tot_length += vad_result.dim

            if not (omit_unvoiced_utts and sum == 0):
                vad_writer.write(utt, vad_result)

        print("Applied energy based voice activity detection; processed " + str(num_done) + " utterances successfully; " +
            str(num_err) + " had empty features, and " + str(num_unvoiced) + " were completely unvoiced.")
        print("Proportion of voiced frames was " + str((tot_decision /
            tot_length)) + " over " + str(tot_length) + " frames.")

        return 0 if num_done != 0 else 1
    except Exception as e:
        print(str(e))
        return -1
