#!/bin/bash

INITIAL_WORKING_DIRECTORY=$(pwd)
cd "$(dirname "$0")"

# create SAD model tree structure
mkdir ../diarizer_model/
mkdir ../diarizer_model/src
mkdir ../diarizer_model/data

# Download and extract SAD system from Chime6 challenge
wget https://kaldi-asr.org/models/12/0012_diarization_v1.tar.gz -P ./
tar xvzf ./0012_diarization_v1.tar.gz
rm ./0012_diarization_v1.tar.gz

cp ./0012_diarization_v1/exp/xvector_nnet_1a/final.raw ../diarizer_model/src/
cp ./0012_diarization_v1/exp/xvector_nnet_1a/plda ../diarizer_model/src/
cp ./0012_diarization_v1/exp/xvector_nnet_1a/max_chunk_size ../diarizer_model/src/
cp ./0012_diarization_v1/exp/xvector_nnet_1a/min_chunk_size ../diarizer_model/src/
cp ./0012_diarization_v1/exp/xvector_nnet_1a/extract.config ../diarizer_model/src/
cp ./0012_diarization_v1/exp/xvector_nnet_1a/mean.vec ../diarizer_model/src/
echo "--vad-energy-threshold=5.5
--vad-energy-mean-scale=0.5" > ../diarizer_model/src/vad.conf

# Clean unnecessary files
rm -r ./0012_diarization_v1

cd $INITIAL_WORKING_DIRECTORY