#!/bin/bash

INITIAL_WORKING_DIRECTORY=$(pwd)
cd "$(dirname "$0")"

# create SAD model tree structure
mkdir ../SAD_model/
mkdir ../SAD_model/src/
mkdir ../SAD_model/data/

echo "" > ../SAD_model/src/wav.scp

# Download and extract SAD system from Chime6 challenge
wget https://kaldi-asr.org/models/12/0012_sad_v1.tar.gz -P ./
tar xvzf ./0012_sad_v1.tar.gz
rm ./0012_sad_v1.tar.gz

cp ./0012_sad_v1/exp/segmentation_1a/tdnn_stats_sad_1a/final.raw ../SAD_model/src/
cp ./0012_sad_v1/exp/segmentation_1a/tdnn_stats_sad_1a/post_output.vec ../SAD_model/src/

# Clean unnecessary files
rm -r ./0012_sad_v1

cd $INITIAL_WORKING_DIRECTORY