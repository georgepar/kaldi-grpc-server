from kaldi.ivector import Plda, PldaConfig
from kaldi import util
from kaldi.util.table import SequentialVectorReader, RandomAccessVectorReader, MatrixWriter, VectorWriter
from kaldi.matrix import Vector, Matrix, DoubleMatrix, DoubleVector
from kaldi.matrix.packed import SpMatrix
from kaldi.base.math import approx_equal
from kaldi.matrix.common import MatrixTransposeType, MatrixResizeType
import numpy
import math


def score_plda(config, target_energy=0.1):
    print("\nInitializing PLDA scoring...")

    ivector_subtract_global_mean("ark:/kaldigrpc/diarizer_model/data/xvectors.ark",
                                 "/kaldigrpc/diarizer_model/data/mean.vec", "ark:/kaldigrpc/diarizer_model/data/subtracted_mean_xvectors.ark")
    transform_vec("/kaldigrpc/diarizer_model/data/transform.mat", "ark:/kaldigrpc/diarizer_model/data/subtracted_mean_xvectors.ark",
                  "ark:/kaldigrpc/diarizer_model/data/transformed_subtracted_mean_xvectors.ark")
    ivector_normalize_length("ark:/kaldigrpc/diarizer_model/data/transformed_subtracted_mean_xvectors.ark",
                             "ark:/kaldigrpc/diarizer_model/data/plda_xvectors.ark")

    print("\nScoring xvectors...")
    ivector_plda_scoring_dense(config.model_cfg.plda, "/kaldigrpc/diarizer_model/data/spk2utt", "ark:/kaldigrpc/diarizer_model/data/plda_xvectors.ark",
                               "ark,scp:/kaldigrpc/diarizer_model/data/scores.ark,/kaldigrpc/diarizer_model/data/scores.scp", target_energy)

    print("\nPLDA scoring complete!")

def ivector_subtract_global_mean(ivector_rspecifier, mean_rxfilename=None, ivector_wspecifier=None):
    usage = '''
    "Copies a table of iVectors but subtracts the global mean as\n"
        "it does so.  The mean may be specified as the first argument; if not,\n"
        "the sum of the input iVectors is used.\n"
        "\n"
        "Usage: ivector-subtract-global-mean <ivector-rspecifier> <ivector-wspecifier>\n"
        "or: ivector-subtract-global-mean <mean-rxfliename> <ivector-rspecifier> <ivector-wspecifier>\n"
        "e.g.: ivector-subtract-global-mean scp:ivectors.scp ark:-\n"
        "or: ivector-subtract-global-mean mean.vec scp:ivectors.scp ark:-\n"
        "See also: ivector-mean\n";
    '''
    try:
        subtract_mean = True
        if ivector_rspecifier is None:
            print(usage)
            exit(1)

        num_done = 0

        if mean_rxfilename is None:
            sum = Vector()
            ivectors = Vector()

            ivector_reader = SequentialVectorReader(ivector_rspecifier)
            ivector_writer = VectorWriter(ivector_wspecifier)

            for key, ivector in ivector_reader:
                if (sum.dim == 0):
                    sum.resize_(ivector.dim)
                sum.add_vec_(1.0, ivector)
                num_done += 1
                ivectors.append([key, Vector(ivector)])

            print("LOG: Read " + str(num_done) + " iVectors.")

            if num_done != 0:
                print("LOG: Norm of iVector mean was " +
                      str(sum.Norm(2.0) / num_done))
                for i in range(ivectors.size()):
                    key = ivectors[i][0]
                    
                    ivector = ivectors[i][1]
                    if subtract_mean:
                        ivector.add_vec_(-1.0 / num_done, sum)
                    ivector_writer.write(key, ivector)
                    ivectors[i][1] = None
        else:
            mean = Vector()
            istream = util.io.Input(mean_rxfilename)
            mean.read_(istream.stream(), binary=False)

            ivector_reader = SequentialVectorReader(ivector_rspecifier)
            ivector_writer = VectorWriter(ivector_wspecifier)
            for key, ivector in ivector_reader:
                ivector.add_vec_(-1.0, mean)
                ivector_writer.write(key, ivector)
                num_done += 1

        print("LOG: Wrote " + str(num_done) + " mean-subtracted iVectors")
        return 0 if num_done != 0 else 1
    except Exception as e:
        print(str(e))
        return -1


def transform_vec(transform_rxfilename, vec_rspecifier, vec_wspecifier):
    '''
        "This program applies a linear or affine transform to individual vectors, e.g.\n"
        "iVectors.  It is transform-feats, except it works on vectors rather than matrices,\n"
        "and expects a single transform matrix rather than possibly a table of matrices\n"
        "\n"
        "Usage: transform-vec [options] <transform-rxfilename> <feats-rspecifier> <feats-wspecifier>\n"
        "See also: transform-feats, est-pca\n"
    '''
    try:
        vec_reader = SequentialVectorReader(vec_rspecifier)
        vec_writer = VectorWriter(vec_wspecifier)

        transform = Matrix()
        istream = util.io.Input(transform_rxfilename)
        transform.read_(istream.stream(), binary=False)

        num_done = 0
        for key, vec in vec_reader:
            transform_rows = transform.num_rows
            transform_cols = transform.num_cols
            vec_dim = vec.dim

            vec_out = Vector(transform_rows)

            if transform_cols == vec_dim:
                vec_out.add_mat_vec_(
                    1.0, transform, MatrixTransposeType.NO_TRANS, vec, 0.0)
            else:
                if transform_cols != vec_dim + 1:
                    print("Error: Dimension mismatch: input vector has dimension " +
                          str(vec.dim) + " and transform has " + str(transform_cols) + " columns.")
                vec_out.copy_col_from_mat_(transform, vec_dim)
                vec_out.add_mat_vec_(1.0, transform.range(
                    0, transform.num_rows, 0, vec_dim), MatrixTransposeType.NO_TRANS, vec, 1.0)

            vec_writer.write(key, vec_out)
            num_done += 1

        print("LOG: Applied transform to " + str(num_done) + " vectors.")

        return 0 if num_done != 0 else 1
    except Exception as e:
        print(str(e))
        return -1


def ivector_normalize_length(ivector_rspecifier, ivector_wspecifier):
    '''
        "Normalize length of iVectors to equal sqrt(feature-dimension)\n"
        "\n"
        "Usage:  ivector-normalize-length [options] <ivector-rspecifier> "
        "<ivector-wspecifier>\n"
        "e.g.: \n"
        " ivector-normalize-length ark:ivectors.ark ark:normalized_ivectors.ark\n"
    '''
    try:
        normalize = True
        scaleup = True
        num_done = 0
        tot_ratio, tot_ratio2 = 0.0, 0.0
        ivector_reader = SequentialVectorReader(ivector_rspecifier)
        ivector_writer = VectorWriter(ivector_wspecifier)

        for key, ivector in ivector_reader:
            norm = ivector.norm(2.0)
            # how much larger it is than it would be, in expectation, if normally
            ratio = norm / math.sqrt(ivector.dim)

            if scaleup == 0:
                ratio = norm

            print("VLOG: Ratio for key " + str(key) + " is " + str(ratio))

            if ratio == 0.0:
                print("WARNING: Zero iVector")
            elif normalize:
                ivector.scale_(1.0 / ratio)

            ivector_writer.write(key, ivector)
            tot_ratio += ratio
            tot_ratio2 += ratio * ratio
            num_done += 1

        print("LOG: Processed " + str(num_done) + " iVectors.")
        if num_done != 0:
            avg_ratio = tot_ratio / num_done
            ratio_stddev = math.sqrt(
                tot_ratio2 / num_done - avg_ratio * avg_ratio)
            print("LOG: Average ratio of iVector to expected length was " +
                  str(avg_ratio) + ", standard deviation was " + str(ratio_stddev))

        return 0 if num_done != 0 else 1
    except Exception as e:
        print(str(e))
        return -1

def ivector_plda_scoring_dense(plda_rxfilename, reco2utt_rspecifier, ivector_rspecifier, scores_wspecifier, target_energy=0.5):
    try:
        plda_config = PldaConfig()
        assert target_energy <= 1.0

        plda = Plda()
        istream = util.io.Input(plda_rxfilename)
        plda.read(istream.stream(), binary=True)

        reco2utt_reader = read_reco2utt(reco2utt_rspecifier)
        ivector_reader = RandomAccessVectorReader(ivector_rspecifier)
        scores_writer = MatrixWriter(scores_wspecifier)

        num_reco_err = 0
        num_reco_done = 0

        for reco, uttlist in reco2utt_reader.items():
            this_plda = Plda().from_other(plda)
            ivectors = []

            for i in range(len(uttlist)):
                utt = uttlist[i]

                if not ivector_reader.has_key(utt):
                    print(
                        "ERROR: No iVector present in input for utterance " + str(utt))

                ivector = ivector_reader.value(utt)
                ivectors.append(ivector)

            if len(ivectors) == 0:
                print("WARNING: Not producing output for recording " +
                      str(reco) + " since no segments had iVectors")
                num_reco_err += 1
            else:
                ivector_mat = Matrix(len(ivectors), ivectors[0].dim)
                ivector_mat_pca, ivector_mat_plda, pca_transform = Matrix(), Matrix(), Matrix()
                scores = Matrix(len(ivectors), len(ivectors))

                for i in range(len(ivectors)):
                    ivector_mat.row(i).copy_(ivectors[i])

                if est_pca_plda(ivector_mat, target_energy, reco, pca_transform):
                    # Apply the PCA transform to the raw i-vectors.
                    apply_pca(ivector_mat, pca_transform, ivector_mat_pca)

                    # Apply the PCA transform to the parameters of the PLDA model.
                    this_plda.apply_transform(DoubleMatrix(pca_transform))

                    # Now transform the i-vectors using the reduced PLDA model.
                    transform_ivectors(
                        ivector_mat_pca, plda_config, this_plda, ivector_mat_plda)

                else:
                    # If EstPca returns false, we won't apply any PCA.
                    transform_ivectors(
                        ivector_mat, plda_config, this_plda, ivector_mat_plda)

                for i in range(ivector_mat_plda.num_rows):
                    for j in range(ivector_mat_plda.num_rows):
                        scores[i][j] = this_plda.log_likelihood_ratio(
                            DoubleVector(ivector_mat_plda.row(i)), 1, DoubleVector(ivector_mat_plda.row(j)))

                scores_writer.write(reco, scores)
                num_reco_done += 1

        print("Processed " + str(num_reco_done) +
              " recordings, " + str(num_reco_err) + " had errors.")
        return 0 if num_reco_done != 0 else 1

    except Exception as e:
        print(str(e))
        return -1


def est_pca_plda(ivector_mat, target_energy, reco, mat):
    if approx_equal(target_energy, 1.0, 0.001):
        return False

    num_rows = ivector_mat.num_rows
    num_cols = ivector_mat.num_cols
    sum = Vector()
    sumsq = SpMatrix()
    sum.resize_(num_cols)
    sumsq.resize_(num_cols)
    sum.add_row_sum_mat_(1.0, ivector_mat)
    sumsq.add_mat2_(1.0, ivector_mat, MatrixTransposeType.TRANS, 1.0)
    sum.scale_(1.0 / num_rows)
    sumsq.scale_(1.0 / num_rows)
    sumsq.add_vec2_(-1.0, sum)    # now sumsq is centered covariance.
    full_dim = sum.dim

    P = Matrix(full_dim, full_dim)
    s = Vector(full_dim)

    try:
        sumsq = Matrix(sumsq)
        u, s, vh = numpy.linalg.svd(sumsq)
        s = Vector(s)
        P = Matrix(u)
    except:
        print("WARNING: Unable to compute conversation dependent PCA for recording " + str(reco) + ".")

    # Transpose of P.  This is what appears in the transform.
    transform = P.transpose_()
    # We want the PCA transform to retain target_energy amount of the total
    # energy.
    total_energy = s.sum()
    energy = 0.0
    dim = 1
    while (energy/total_energy) <= target_energy:
        energy += s[dim-1]
        dim += 1
    transform_float = Matrix(transform)

    mat.resize_(transform.num_cols, transform.num_rows)
    mat.copy_(transform)
    mat.resize_(dim, transform_float.num_cols, MatrixResizeType.COPY_DATA)

    return True


def transform_ivectors(ivectors_in, plda_config, plda, ivectors_out):
    '''
    Transforms i-vectors using the PLDA model.
    '''
    dim = plda.dim()
    ivectors_out.resize_(ivectors_in.num_rows, dim)

    for i in range(ivectors_in.num_rows):
        transformed_ivector = DoubleVector(dim)
        plda.transform_ivector(plda_config, DoubleVector(
            ivectors_in.row(i)), 1, transformed_ivector)
        ivectors_out.row(i).copy_(transformed_ivector)

    return


def apply_pca(ivectors_in, pca_mat, ivectors_out):
    '''
    Transform the i-vectors using the recording-dependent PCA matrix.
    '''
    transform_cols = pca_mat.num_cols
    transform_rows = pca_mat.num_rows
    feat_dim = ivectors_in.num_cols
    ivectors_out.resize_(ivectors_in.num_rows, transform_rows)
    assert transform_cols == feat_dim
    ivectors_out.add_mat_mat_(
        ivectors_in, pca_mat, MatrixTransposeType.NO_TRANS, MatrixTransposeType.TRANS, 1.0, 0.0)

def read_reco2utt(reco2utt):
    '''
    Reads a reco2utt (spk2utt) file and returns a dictionary of it
    '''

    myDict = {}
    with open(reco2utt) as f:
        lines = f.read().splitlines()

    for i in lines:
        temp = i.split(" ", 1)
        reco = temp[0]
        utterances = [s.replace("[", "").replace("]", "").replace("/", "").strip() for s in temp[1].split(" ")]

        if reco not in myDict:
            myDict[reco] = utterances
        else:
            myDict[reco].extend(utterances)

    return myDict