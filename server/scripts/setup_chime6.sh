#!/bin/sh

INITIAL_WORKING_DIRECTORY=$(pwd)
cd "$(dirname "$0")"

# create model tree structure
mkdir ../chime6_model/
mkdir ../chime6_model/ivector_extractor/
mkdir ../chime6_model/conf/

echo "" > ../chime6_model/ivector_extractor/online_cmvn_iextractor
echo "--cmvn-config=conf/online_cmvn.conf
--splice-config=conf/splice.conf
--diag-ubm=ivector_extractor/final.dubm
--ivector-extractor=ivector_extractor/final.ie
--lda-matrix=ivector_extractor/final.mat
--global-cmvn-stats=ivector_extractor/global_cmvn.stats" > ../chime6_model/conf/ivector_extractor.conf
echo "--use-energy=false   # use average of log energy, not energy.
--sample-frequency=16000 
--num-mel-bins=40
--num-ceps=40
--low-freq=40
--high-freq=-400" > ../chime6_model/conf/mfcc.conf
echo "# configuration file for apply-cmvn-online, used in the script ../local/online/run_online_decoding_nnet2.sh" > ../chime6_model/conf/online_cmvn.conf
echo "--frame-subsampling-factor=3
--acoustic-scale=1.0 
--feature-type=mfcc
--mfcc-config=conf/mfcc.conf
--ivector-extraction-config=conf/ivector_extractor.conf" > ../chime6_model/conf/online.conf
echo "--left-context=3
--right-context=3" > ../chime6_model/conf/splice.conf

# Download and extract Chime6 from kaldi official site
wget https://kaldi-asr.org/models/12/0012_asr_v1.tar.gz -P ./
tar xvzf ./0012_asr_v1.tar.gz
rm ./0012_asr_v1.tar.gz

# Parse i-vector extractor files  
cp ./0012_asr_v1/exp/nnet3_train_worn_simu_u400k_cleaned_rvb/extractor/* ../chime6_model/ivector_extractor/
rm ../chime6_model/ivector_extractor/final.ie.id

# Parse data files (words)
cp ./0012_asr_v1/data/lang/words.txt ../chime6_model/

# Parse HCLG
cp ./0012_asr_v1/exp/chain_train_worn_simu_u400k_cleaned_rvb/tree_sp/graph/HCLG.fst ../chime6_model/

# Parse main nn model (.mdl)
cp ./0012_asr_v1/exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/final.mdl ../chime6_model/

# Parse global cmvn stats
cp ./0012_asr_v1/exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/cmvn_opts ../chime6_model/
mv ../chime6_model/cmvn_opts ../chime6_model/global_cmvn.stats

# Clean unnecessary files
rm -r ./0012_asr_v1

cd $INITIAL_WORKING_DIRECTORY