import sys
import subprocess
import os

from kaldi.util.table import SequentialMatrixReader, IntWriter, RandomAccessIntReader
from kaldi.ivector import AgglomerativeClusterer
from .plda import read_reco2utt
from .make_rttm import create_rttm


def apply_clustering(config, rttm_channel=1):
    print("Performing clustering using PLDA scores...")

    os.system(
        "awk '{print $1, '4'}' /kaldigrpc/diarizer_model/data/wav.scp > /kaldigrpc/diarizer_model/data/reco2num_spk")

    print("Starting agglomerative clustering...")
    agglomerative_cluster("scp:/kaldigrpc/diarizer_model/data/scores.scp",
                          "/kaldigrpc/diarizer_model/data/spk2utt",
                          "ark,t:/kaldigrpc/diarizer_model/data/labels",
                          "ark,t:/kaldigrpc/diarizer_model/data/reco2num_spk", threshold = 0.4)

    print("Computing RTTM...")
    create_rttm("/kaldigrpc/diarizer_model/data/filtered_segments", "/kaldigrpc/diarizer_model/data/labels", "/kaldigrpc/diarizer_model/data/rttm", rttm_channel)

    print("Clustering complete!")

def agglomerative_cluster(scores_rspecifier, reco2utt_rspecifier, label_wspecifier, reco2num_spk_rspecifier=None, threshold=0.0, max_spk_fraction=1.0, read_costs=False, first_pass_max_utterances=32767):
    try:
        usage = '''
        "Cluster utterances by similarity score, used in diarization.\n"
        "Takes a table of score matrices indexed by recording, with the\n"
        "rows/columns corresponding to the utterances of that recording in\n"
        "sorted order and a reco2utt file that contains the mapping from\n"
        "recordings to utterances, and outputs a list of labels in the form\n"
        "<utt> <label>.  Clustering is done using agglomerative hierarchical\n"
        "clustering with a score threshold as stop criterion.  By default, the\n"
        "program reads in similarity scores, but with --read-costs=true\n"
        "the scores are interpreted as costs (i.e. a smaller value indicates\n"
        "utterance similarity).\n"
        "Usage: agglomerative-cluster [options] <scores-rspecifier> "
        "<reco2utt-rspecifier> <labels-wspecifier>\n"
        "e.g.: \n"
        " agglomerative-cluster ark:scores.ark ark:reco2utt \n"
        "   ark,t:labels.txt\n"
            '''

        scores_reader = SequentialMatrixReader(scores_rspecifier)
        reco2utt_reader = read_reco2utt(reco2utt_rspecifier)
        reco2num_spk_reader = RandomAccessIntReader(reco2num_spk_rspecifier)
        label_writer = IntWriter(label_wspecifier)

        if read_costs == 0:
            threshold = -threshold

        for reco, costs in scores_reader:
            # By default, the scores give the similarity between pairs of
            # utterances.  We need to multiply the scores by -1 to reinterpet
            # them as costs (unless --read-costs=true) as the agglomerative
            # clustering code requires.

            if read_costs == 0:
                costs.scale_(-1)

            uttlist = reco2utt_reader[reco]
            spk_ids = []

            if reco2num_spk_rspecifier is not None:
                num_speakers = reco2num_spk_reader.value(reco)

                if 1.0 / num_speakers <= max_spk_fraction and max_spk_fraction <= 1.0:
                    spk_ids = AgglomerativeClusterer(costs, sys.float_info.max,
                                                     num_speakers, first_pass_max_utterances, max_spk_fraction).cluster()
                else:
                    spk_ids = AgglomerativeClusterer(costs, sys.float_info.max,
                                                     num_speakers, first_pass_max_utterances, 1.0).cluster()
            else:
                spk_ids = AgglomerativeClusterer(
                    costs, threshold, 1, first_pass_max_utterances, 1.0).cluster()

            for i in range(len(spk_ids)):
                label_writer.write(uttlist[i], spk_ids[i])

        return 0
    except Exception as e:
        print(str(e))
        return -1
