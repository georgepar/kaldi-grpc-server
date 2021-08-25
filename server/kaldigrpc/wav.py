from typing import Iterator

import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

from kaldigrpc.config import AudioConfig
from kaldi.matrix import SubVector


def pydub2kaldi(sound: AudioSegment) -> SubVector:
    channel_sounds = sound.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr = fp_arr.reshape(-1)

    wav = SubVector(fp_arr)

    return wav


def load_wav_file(wav_file: str) -> SubVector:
    snd = AudioSegment.from_file(wav_file)
    wav = pydub2kaldi(snd)

    return wav


def bytes2vector(wavbytes: bytes, config: AudioConfig) -> SubVector:
    sample_width = config.sample_width
    snd = AudioSegment(
        wavbytes,
        sample_width=sample_width,
        frame_rate=config.frame_rate,
        channels=config.channels,
    )
    wav = pydub2kaldi(snd)

    return wav


def load_chunked_wav(wav_file: str, chunk_size: int = 1440) -> Iterator[SubVector]:
    snd = AudioSegment.from_file(wav_file)
    chunks = make_chunks(snd, chunk_size)

    if snd.sample_width == 2:
        encoding = "LINEAR16"
    else:
        encoding = "MULAW"

    config = AudioConfig(snd.frame_rate, encoding, 1)

    for chunk in chunks:
        yield bytes2vector(chunk.raw_data, config)
