import os
from dataclasses import dataclass
from typing import Optional

from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions, NnetSimpleComputationOptions
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo)
from kaldi.segmentation import NnetSAD, SegmentationProcessor
from kaldi.util.options import ParseOptions


@dataclass
class AudioConfig:
    frame_rate: float = 16000.0
    encoding: str = "LINEAR16"
    channels: int = 1

    @property
    def sample_width(self) -> int:
        if self.encoding == "LINEAR16":
            return 2
        elif self.encoding == "MULAW":
            return 1
        else:
            raise ValueError(f"Encoding {self.encoding} is not supported")

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_float("frame_rate", 16000.0, "Accepted wav frame rate")
        po.register_str(
            "encoding",
            "LINEAR16",
            "Wav encoding: [LINEAR16: linear 16-bit pcm, MULAW: 8-bit PCM]",
        )
        po.register_int("channels", 1, "Number of input wav channels")

    @classmethod
    def from_opts(cls, po: ParseOptions):
        opts = po._get_options()

        return cls(
            opts.float_map["frame_rate"],
            opts.str_map["encoding"],
            opts.int_map["channels"],
        )


@dataclass
class ModelConfig:
    finalmdl: str = "/model/final.mdl"
    hclgfst: str = "/model/HCLG.fst"
    wordstxt: str = "/model/words.txt"

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("model_dir", "/model", "Path to kaldi model")

    @classmethod
    def from_opts(cls, po: ParseOptions):
        opts = po._get_options()
        model_dir = opts.str_map["model_dir"]

        return cls(
            os.path.join(model_dir, "final.mdl"),
            os.path.join(model_dir, "HCLG.fst"),
            os.path.join(model_dir, "words.txt"),
        )


@dataclass
class RnnLmConfig:
    use_rnnlm: bool
    old_lm: str
    word_feats: str
    vocab: str
    feat_embedding: str
    rnnlm_model: str
    bos: str
    eos: str
    brk: str

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_bool("use_rnnlm", False, "Use RNNLM rescoring")
        po.register_str(
            "rnnlm_config", "/model/conf/rnnlm.conf", "Path to rnnlm config file"
        )
        po.register_str("old_lm", "/model/rnnlm/G.fst", "Old fst language model")
        po.register_str("word_feats", "/model/rnnlm/word_feats.fst", "Word feats file")
        po.register_str(
            "feat_embedding", "/model/rnnlm/feat_embedding.final.mat", "Feat embedding"
        )
        po.register_str("rnnlm_model", "/model/rnnlm/final.raw", "RNNLM trained model")
        po.register_str(
            "vocab", "/model/rnnlm/config/words.txt", "RNNLM vocabulary file"
        )
        po.register_str("bos", "<s>", "BOS token")
        po.register_str("eos", "</s>", "EOS token")
        po.register_str("brk", "<brk>", "BRK token")

    @classmethod
    def from_opts(cls, po: ParseOptions):
        rnnlm_config = po._get_options().str_map["rnnlm_config"]

        if os.path.isfile(rnnlm_config):
            po.read_config_file(rnnlm_config)

        opts = po._get_options()

        return cls(
            opts.bool_map["use_rnnlm"],
            opts.str_map["old_lm"],
            opts.str_map["word_feats"],
            opts.str_map["vocab"],
            opts.str_map["feat_embedding"],
            opts.str_map["rnnlm_model"],
            opts.str_map["bos"].replace('"', "").replace("'", ""),
            opts.str_map["eos"].replace('"', "").replace("'", ""),
            opts.str_map["brk"].replace('"', "").replace("'", ""),
        )


@dataclass
class AsrConfig:
    feat: OnlineNnetFeaturePipelineInfo
    endpoint: OnlineEndpointConfig
    decoder: LatticeFasterDecoderOptions
    decodable: NnetSimpleLoopedComputationOptions
    audio: AudioConfig
    model: ModelConfig
    rnnlm: RnnLmConfig

    @classmethod
    def parse_options(cls, po: Optional[ParseOptions]):
        RnnLmConfig.register(po)
        ModelConfig.register(po)
        AudioConfig.register(po)
        feat_opts = OnlineNnetFeaturePipelineConfig()
        endpoint_opts = OnlineEndpointConfig()
        decoder_opts = LatticeFasterDecoderOptions()
        decoder_opts.beam = 13
        decoder_opts.max_active = 10000
        decodable_opts = NnetSimpleLoopedComputationOptions()
        decodable_opts.acoustic_scale = 1.0
        decodable_opts.frame_subsampling_factor = 3
        decodable_opts.frames_per_chunk = 150
        feat_opts.register(po)
        endpoint_opts.register(po)
        decoder_opts.register(po)
        decodable_opts.register(po)
        args = po.parse_args()
        po.read_config_file(os.path.join(args.model_dir, "conf", "online.conf"))
        feat_info = OnlineNnetFeaturePipelineInfo.from_config(feat_opts)

        print("Running with the user provided configuration")
        audio_cfg = AudioConfig.from_opts(po)
        model_cfg = ModelConfig.from_opts(po)
        rnnlm_cfg = RnnLmConfig.from_opts(po)
        po.print_config()

        cfg = cls(
            feat_info,
            endpoint_opts,
            decoder_opts,
            decodable_opts,
            audio_cfg,
            model_cfg,
            rnnlm_cfg,
        )

        return cfg, args

@dataclass
class SadModelConfig:
    finalraw: str = "/kaldigrpc/SAD_model/src"
    postoutput: str = "/kaldigrpc/SAD_model/src"
    wavscp: str = "/kaldigrpc/SAD_model/data"

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("model_dir", "/kaldigrpc/SAD_model", "Path to SAD model")

    @classmethod
    def from_opts(cls, po: ParseOptions):
        opts = po._get_options()
        model_dir = opts.str_map["model_dir"]

        return cls(
            os.path.join(model_dir, "src/final.raw"),
            os.path.join(model_dir, "src/post_output.vec"),
            os.path.join(model_dir, "data/wav.scp")
        )

@dataclass
class SadConfig:
    transform: NnetSAD
    graph: NnetSAD
    sad: NnetSAD
    seg: SegmentationProcessor
    decodable: NnetSimpleComputationOptions
    model: SadModelConfig
    feats_rspec: str

    @classmethod
    def parse_options(cls, po: Optional[ParseOptions]):
        SadModelConfig.register(po)
        model_cfg = SadModelConfig.from_opts(po)

        model = NnetSAD.read_model(model_cfg.finalraw)
        post = NnetSAD.read_average_posteriors(model_cfg.postoutput)
        transform = NnetSAD.make_sad_transform(post)
        graph = NnetSAD.make_sad_graph()
        decodable_opts = NnetSimpleComputationOptions()
        decodable_opts.extra_left_context = 79
        decodable_opts.extra_right_context = 21
        decodable_opts.extra_left_context_initial = -1
        decodable_opts.extra_right_context_final = -1
        decodable_opts.frames_per_chunk = 150
        decodable_opts.acoustic_scale = 1 # original value from kaldi was 0.3

        sad = NnetSAD(model, transform, graph, decodable_opts=decodable_opts)
        seg = SegmentationProcessor(target_labels=[2])

        feats_rspec = "ark:compute-mfcc-feats --config=/kaldigrpc/model/conf/mfcc.conf scp:{} ark:- |".format(model_cfg.wavscp)
        decodable_opts.register(po)
        args = po.parse_args()
        
        po.print_config()

        cfg = cls(
            transform,
            graph,
            sad,
            seg,
            decodable_opts,
            model_cfg,
            feats_rspec
        )

        return cfg, args

@dataclass
class DiarizerModelConfig:
    finalraw: str = "/kaldigrpc/diarizer_model/src/final.raw"
    plda: str = "/kaldigrpc/diarizer_model/src/plda"
    max_chunk_size: str = "/kaldigrpc/diarizer_model/src/max_chunk_size"
    min_chunk_size: str = "/kaldigrpc/diarizer_model/src/min_chunk_size"
    extractconfig: str = "/kaldigrpc/diarizer_model/src/extract.config"
    segments: str = "/kaldigrpc/diarizer_model/data/segments"
    featsscp: str = "/kaldigrpc/diarizer_model/data/feats.scp"

    @classmethod
    def register(cls, po: ParseOptions):
        po.register_str("model_dir", "/kaldigrpc/diarizer_model", "Path to Diarizer model")

    @classmethod
    def from_opts(cls, po: ParseOptions):
        opts = po._get_options()
        model_dir = opts.str_map["model_dir"]

        return cls(
            os.path.join(model_dir, "src/final.raw"),
            os.path.join(model_dir, "src/plda"),
            os.path.join(model_dir, "src/max_chunk_size"),
            os.path.join(model_dir, "src/min_chunk_size"),
            os.path.join(model_dir, "src/extract.config"),
            os.path.join(model_dir, "data/segments"),
            os.path.join(model_dir, "data/feats.scp")
        )

@dataclass
class DiarizerConfig:
    model_cfg: DiarizerModelConfig
    feats_rspec: str
    hard_min: bool
    min_segment: float
    window: float
    period: float
    min_chunk_size: int
    nnet: str
    pca_dim: any

    @classmethod
    def parse_options(cls, po: Optional[ParseOptions]):

        #TODO Whole config needs cleaning and structuring

        DiarizerModelConfig.register(po)
        model_cfg = DiarizerModelConfig.from_opts(po)

        chunk_size=-1 # The chunk size over which the embedding is extracted.
              # If left unspecified, it uses the max_chunk_size in the nnet
              # directory.
        window=1.5
        period=0.75
        pca_dim=None
        min_segment=0.5
        hard_min=False
        apply_cmn=True # If true, apply sliding window cepstral mean normalization
        max_chunk_size = 0
        min_chunk_size = 0
        nnet = ""

        #TODO Check if input files exist
        
        with open(model_cfg.max_chunk_size) as f:
            max_chunk_size = int(f.readlines()[0].strip())
        with open(model_cfg.min_chunk_size) as f:
            min_chunk_size = int(f.readlines()[0].strip())

        #TODO this is unnecessary probably
        if os.path.exists(model_cfg.extractconfig):
            nnet= "nnet3-copy --nnet-config={} {} - |".format(model_cfg.extractconfig, model_cfg.finalraw)
        else:
            nnet = model_cfg.finalraw

        if chunk_size <= 0:
            chunk_size = max_chunk_size
        if max_chunk_size < chunk_size:
            print("Specified chunk size of {} is larger than the maximum chunk size, {}".format(chunk_size, max_chunk_size))
        
        #TODO this is unnecessary probably
        if apply_cmn:
            feats_rspec = "ark,s,cs:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:{} ark:- |".format(model_cfg.featsscp)
        else:
            feats_rspec="scp:{}".format(model_cfg.featsscp)

        args = po.parse_args()
        po.print_config()

        cfg = cls(
            model_cfg,
            feats_rspec,
            hard_min,
            min_segment,
            window,
            period,
            min_chunk_size,
            nnet,
            pca_dim
        )

        return cfg, args

if __name__ == "__main__":
    cfg = AsrConfig.parse_options()
    print(cfg)
