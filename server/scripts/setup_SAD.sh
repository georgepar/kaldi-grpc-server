#!/bin/bash

INITIAL_WORKING_DIRECTORY=$(pwd)
cd "$(dirname "$0")"

# create SAD model tree structure
mkdir ../SAD_model/

echo "# config for high-resolution MFCC features, intended for neural network training.
# Note: we keep all cepstra, so it has the same info as filterbank features,
# but MFCC is more easily compressible (because less correlated) which is why
# we prefer this method.
--use-energy=false   # use average of log energy, not energy.
--sample-frequency=16000 
--num-mel-bins=40
--num-ceps=40
--low-freq=40
--high-freq=-400" > ../SAD_model/mfcc.conf

echo "" > ../SAD_model/wav.scp

# Download and extract SAD system from Chime6 challenge
wget https://kaldi-asr.org/models/12/0012_sad_v1.tar.gz -P ./
tar xvzf ./0012_sad_v1.tar.gz
rm ./0012_sad_v1.tar.gz

cp ./0012_sad_v1/exp/segmentation_1a/tdnn_stats_sad_1a/final.raw ../SAD_model/
cp ./0012_sad_v1/exp/segmentation_1a/tdnn_stats_sad_1a/post_output.vec ../SAD_model/

# Clean unnecessary files
rm -r ./0012_sad_v1

cd $INITIAL_WORKING_DIRECTORY