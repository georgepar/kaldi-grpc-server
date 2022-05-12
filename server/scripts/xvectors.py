import math
import numpy as np

from kaldi import nnet3, util
from kaldi.util.table import MatrixWriter, SequentialMatrixReader, SequentialMatrixReader, SequentialVectorReader, RandomAccessVectorReader, IntWriter, VectorWriter
from kaldi.nnet3 import (NnetComputer, ComputationRequest, IoSpecification, NnetComputeOptions,
                         NnetSimpleComputationOptions, CachingOptimizingCompilerOptions, CachingOptimizingCompiler, CollapseModelConfig)
from kaldi.cudamatrix import CuMatrix
from kaldi.matrix import Vector, DoubleVector, Matrix, DoubleMatrix, SubMatrix, functions
from kaldi.matrix.packed import DoubleSpMatrix
from kaldi.matrix.common import MatrixTransposeType, MatrixResizeType
from kaldi.nnet3._nnet_common import Index


def extract_vectors(config, apply_cmn=True, window = None):
    print("\nStarting x-vector extraction...")
    feats_rspecifier = "ark:../diarizer_model/data/feats.ark"

    # STAGE 0: Set up new xvector features if APPLY CMVN
    if apply_cmn:
        rspecifier = (
            "ark:compute-mfcc-feats --config=/kaldigrpc/model/conf/mfcc.conf scp:/kaldigrpc/diarizer_model/data/wav2.scp ark:-"
            " | apply-cmvn-sliding --norm-vars=false --cmn-window={} --center=true ark:- ark:-"
            " | select-voiced-frames ark:- scp,s,cs:/kaldigrpc/diarizer_model/data/vad.scp ark:- |"
        ).format(window)

        with MatrixWriter("ark,scp:../diarizer_model/data/xvectors_cvmn_feats.ark,../diarizer_model/data/xvectors_cvmn_feats.scp") as writer:
            for key, feats in SequentialMatrixReader(rspecifier):
                writer[key] = feats

            feats_rspecifier = "ark:../diarizer_model/data/xvectors_cmvn_feats.ark"

    # STAGE 1: NNET3-xvector-compute
    print("Extracting xvectors from nnet...")
    NnetXvectorCompute(
        config, feats_rspecifier, "ark,scp:../diarizer_model/data/xvectors.ark,../diarizer_model/data/xvectors.scp")

    # STAGE 3: Computing mean of x-vectors
    print("\nComputing mean of xvectors...")
    iVectorMean("ark:/kaldigrpc/diarizer_model/data/xvectors.ark",
                "/kaldigrpc/diarizer_model/data/mean.vec")

    # STAGE 4: Computing whitening transform
    if config.pca_dim is None:
        config.pca_dim = -1
    print("\nComputing whitening transform...")
    estPca("ark:/kaldigrpc/diarizer_model/data/xvectors.ark",
           "/kaldigrpc/diarizer_model/data/transform.mat", read_vectors=True, normalize_mean=False, normalize_variance=True, dim=config.pca_dim, binary=False)
    print("X-vector extraction complete!")

def NnetXvectorCompute(config, feature_rspecifier, vector_wspecifier):
    opts = NnetSimpleComputationOptions()
    compiler_config = CachingOptimizingCompilerOptions()

    opts.acoustic_scale = 1.0  # by default do no scaling in this recipe.
    chunk_size = -1
    min_chunk_size = 100
    pad_input = True

    istream = util.io.Input(config.model_cfg.finalraw)
    nnet = nnet3.Nnet().read(istream.stream(), binary=True)
    istream2 = util.io.Input(config.model_cfg.extractconfig)
    nnet.read_config(istream2.stream())

    nnet3.set_batchnorm_test_mode(True, nnet)
    nnet3.set_dropout_test_mode(True, nnet)
    nnet3.collapse_model(CollapseModelConfig(), nnet)

    compiler = CachingOptimizingCompiler.new_with_optimize_opts(
        nnet, opts.optimize_config, compiler_config)

    vector_writer = VectorWriter(vector_wspecifier)

    num_success = 0
    num_fail = 0
    frame_count = 0
    xvector_dim = nnet.output_dim("output")

    feature_reader = SequentialMatrixReader(feature_rspecifier)
    for utt, features in feature_reader:
        if features.num_rows == 0:
            print("Zero-length utterance: " + str(utt))
            num_fail += 1

        num_rows = features.num_rows
        feat_dim = features.num_cols
        this_chunk_size = chunk_size

        if not pad_input and (num_rows < min_chunk_size):
            print("Minimum chunk size of " + str(min_chunk_size) +
                  " is greater than the number of rows in utterance: " + str(utt))
            num_fail += 1
        elif num_rows < chunk_size:
            print("Chunk size of " + str(chunk_size) + " is greater than the number of rows in utterance: " +
                  str(utt) + ", using chunk size  of " + str(num_rows))
            this_chunk_size = num_rows
        elif chunk_size == -1:
            this_chunk_size = num_rows

        num_chunks = math.ceil(num_rows / float(this_chunk_size))
        xvector_avg = Vector(xvector_dim)
        tot_weight = 0.0

        # Iterate over the feature chunks.
        for chunk_indx in range(0, num_chunks):
            # If we're nearing the end of the input, we may need to shift the
            # offset back so that we can get this_chunk_size frames of input to
            # the nnet.
            offset = int(min(this_chunk_size, num_rows -
                         chunk_indx * this_chunk_size))
            if not pad_input and (offset < min_chunk_size):
                continue
            sub_features = SubMatrix(
                features, chunk_indx * this_chunk_size, offset, 0, feat_dim)
            xvector = Vector()
            tot_weight += offset

            # Pad input if the offset is less than the minimum chunk size
            if pad_input and (offset < min_chunk_size):
                padded_features = Matrix(min_chunk_size, feat_dim)
                left_context = int((min_chunk_size - offset) / 2)
                right_context = int(min_chunk_size - offset - left_context)
                for i in range(0, offset):
                    padded_features.row(i).copy_(sub_features.row(0))
                for i in range(0, right_context):
                    padded_features.row(min_chunk_size - i - 1).copy_(sub_features.row(offset - 1))

                padded_features.range(
                    left_context, offset, 0, feat_dim).copy_(sub_features)
                RunNnetComputation(padded_features, nnet, compiler, xvector)
            else:
                RunNnetComputation(sub_features, nnet, compiler, xvector)
            xvector_avg.add_vec_(offset, xvector)

        xvector_avg.scale_(1.0 / tot_weight)
        vector_writer.write(utt, xvector_avg)

        frame_count += features.num_rows
        num_success += 1


def iVectorMean(ivector_rspecifier, mean_wxfilename=None, spk2utt_rspecifier=None, num_utts_wspecifier=None, ivector_wspecifier=None):
    # TODO Second part of if statement is not used/debugged yet
    if(mean_wxfilename is not None):
        num_done = 0
        try:
            ivector_reader = SequentialVectorReader(ivector_rspecifier)
        except IOError:
            exit(1)

        sum = Vector()
        for _, value in ivector_reader:
            if (sum.dim == 0):
                sum.resize_(value.shape[0])
            sum.add_vec_(1.0, value)
            num_done += 1

        if (num_done == 0):
            print("ERROR: No iVectors read")
        else:
            sum.scale_(1.0 / num_done)
            istream = util.io.Output(mean_wxfilename, binary=False)
            sum.write(istream.stream(), binary=False)

            return 0
    else:
        spk_sumsq = 0.0
        spk_sum = Vector()

        num_spk_done = 0
        num_spk_err = 0
        num_utt_done = 0
        num_utt_err = 0

        ivector_reader = RandomAccessVectorReader(ivector_rspecifier)
        spk2utt_reader = SequentialVectorReader(spk2utt_rspecifier)
        ivector_writer = VectorWriter(ivector_wspecifier)
        num_utts_writer = IntWriter(num_utts_wspecifier)

        for spk, uttlist in spk2utt_reader:
            if not uttlist:
                print("Speaker with no utterances.")

            spk_mean = Vector()
            utt_count = 0
            for i in uttlist:
                utt = uttlist[i]
                if not ivector_reader.has_key(utt):
                    print("No iVector present in input for utterance " + str(utt))
                    num_utt_err += 1
                else:
                    if utt_count == 0:
                        spk_mean = ivector_reader.value(utt)
                    else:
                        spk_mean.add_vec_(1.0, ivector_reader.value(utt))

                    num_utt_done += 1
                    utt_count += 1

            if utt_count == 0:
                print("Not producing output for speaker " +
                      str(spk) + " since no utterances had iVectors")
                num_spk_err += 1
            else:
                spk_mean.scale(1.0 / utt_count)
                ivector_writer.write(spk, spk_mean)
                if num_utts_wspecifier != "":
                    num_utts_writer.write(spk, utt_count)
                num_spk_done += 1
                spk_sumsq += functions.vec_vec(spk_mean, spk_mean)
                if spk_sum.dim() == 0:
                    spk_sum.resize(spk_mean.dim())
                spk_sum.add_vec_(1.0, spk_mean)

        print("Computed mean of " + str(num_spk_done) + " speakers ("
              + str(num_spk_err) + " with no utterances), consisting of "
              + str(num_utt_done) + " utterances (" + str(num_utt_err)
              + " absent from input).")

        if num_spk_done != 0:
            spk_sumsq /= num_spk_done
            spk_sum.scale(1.0 / num_spk_done)
            mean_length = spk_sum.norm(2.0)
            spk_length = math.sqrt(spk_sumsq)
            norm_spk_length = spk_length / math.sqrt(spk_sum.dim())

            print("Norm of mean of speakers is " + str(mean_length)
                  + ", root-mean-square speaker-iVector length divided by "
                  + "sqrt(dim) is " + str(norm_spk_length))

        return 0 if num_spk_done != 0 else 1


def estPca(rspecifier, pca_mat_wxfilename, binary=True, read_vectors=False, normalize_variance=False, normalize_mean=False, dim=-1, full_matrix_wxfilename=""):
    num_done = 0
    num_err = 0
    count = 0
    sum = DoubleVector()
    sumsq = DoubleSpMatrix()

    if not read_vectors:
        feat_reader = SequentialMatrixReader(rspecifier)

        for _ in feat_reader:
            mat = Matrix(feat_reader.value())
            if (mat.num_rows == 0):
                print("Empty feature matrix")
                num_err += 1
                continue

            if sum.dim() == 0:
                sum.resize(mat.num_cols())
                sumsq.resize(mat.num_cols())

            if sum.dim() != mat.num_cols():
                print("Feature dimension mismatch " +
                      str(sum.Dim()) + " vs. " + str(mat.num_cols()))
                num_err += 1
                continue

            sum.add_row_sum_mat_(1.0, mat)
            sumsq.add_mat2_(1.0, mat, MatrixTransposeType.TRANS, 1.0)
            count += mat.num_rows
            num_done += 1

        print("Accumulated stats from " + str(num_done) + " feature files, " +
              str(num_err) + " with errors; " + str(count) + " frames.")
    else:
        vec_reader = SequentialVectorReader(rspecifier)

        for _, value in vec_reader:
            vec = DoubleVector(value)
            if (vec.dim == 0):
                print("Warning: Empty input vector")
                num_err += 1
                continue

            if (sum.dim == 0):
                sum.resize_(vec.dim)
                sumsq.resize_(vec.dim)

            if (sum.dim != vec.dim):
                print("Warning: Feature dimension mismatch " +
                      str(sum.dim) + " vs. " + str(vec.dim))
                num_err += 1
                continue

            sum.add_vec_(1.0, vec)
            sumsq.add_vec2_(1.0, vec)

            count += 1.0
            num_done += 1

        print("Accumulated stats from " + str(num_done) +
              " vectors, " + str(num_err) + " with errors.")

    if num_done == 0:
        print("Error: No data accumulated.")

    sum.scale_(1.0 / count)
    sumsq.scale_(1.0 / count)

    sumsq.add_vec2_(-1, sum)  # now sumsq is centered covariance.

    full_dim = sum.dim
    if dim <= 0:
        dim = full_dim
    if dim > full_dim:
        print("Final dimension " + str(dim) +
              " is greater than feature " + "dimension " + str(full_dim))

    # Instead of sumsq.Eig(&s, &P); and sort_svd:
    sumsq = DoubleMatrix(sumsq)
    u, s, _ = np.linalg.svd(sumsq)
    s = DoubleVector(s)
    P = DoubleMatrix(u)

    print("Log: Eigenvalues in PCA are " + str(s))
    print("Log: Sum of PCA eigenvalues is " + str(s.sum()) +
          ", sum of kept eigenvalues is " + str(s.range(0, dim).sum()))

    # Transpose of P.  This is what appears in the transform.
    transform = P.transpose_()

    if (normalize_variance):
        for i in range(full_dim):
            this_var = s[i]
            min_var = 1.0e-15
            if this_var < min_var:
                print("Warning: --normalize-variance option: very tiny variance " +
                      str(s[i]) + "encountered, treating as " + str(min_var))
                this_var = min_var
            # scale on features that will make the variance unit.
            scale = 1.0 / math.sqrt(this_var)
            transform.row(i).scale_(scale)

    offset = DoubleVector(full_dim)

    if (normalize_mean):
        offset.add_mat_vec_(-1.0, transform,
                            MatrixTransposeType.NO_TRANS, sum, 0.0)
        # Add column to transform.
        transform.resize_(full_dim, full_dim + 1, MatrixResizeType.COPY_DATA)
        transform.copy_col_from_vec_(offset, full_dim)

    transform_float = Matrix(transform)

    if (full_matrix_wxfilename != ""):
        istream = util.io.Output(full_matrix_wxfilename, binary)
        transform_float.write(istream.stream(), binary)

    transform_float.resize_(dim, transform_float.num_cols,
                            MatrixResizeType.COPY_DATA)

    istream = util.io.Output(pca_mat_wxfilename, binary)
    transform_float.write(istream.stream(), binary)

    return 0


def RunNnetComputation(features, nnet, compiler, xvector):
    request = ComputationRequest()
    request.need_model_derivative = False
    request.store_component_stats = False
    request.inputs = [IoSpecification().from_interval(
        "input", 0, features.num_rows)]

    request.outputs = [IoSpecification().from_indexes(
        "output", [Index()], False)]

    # Note: compile() returns a NnetComputation object
    computation = compiler.compile(request)
    computer = NnetComputer(NnetComputeOptions(), computation, nnet, None)
    input_feats_cu = CuMatrix().from_matrix(features)
    computer.accept_input("input", input_feats_cu)
    computer.run()

    cu_output = CuMatrix
    cu_output = computer.get_output_destructive("output")
    xvector.resize_(cu_output.num_cols())

    temp = Matrix(cu_output.num_rows(), cu_output.num_cols())
    cu_output.copy_to_mat(temp)
    xvector.copy_row_from_mat_(temp, 0)

    return xvector
