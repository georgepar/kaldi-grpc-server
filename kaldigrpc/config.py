import argparse
import os
from dataclasses import dataclass
from typing import Optional

from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo)
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


if __name__ == "__main__":
    cfg = AsrConfig.parse_options()
    print(cfg)
