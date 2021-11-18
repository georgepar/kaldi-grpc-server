#!/bin/sh

# Download and extract Chime6 from kaldi official site
wget https://kaldi-asr.org/models/12/0012_asr_v1.tar.gz
tar xvzf 0012_asr_v1.tar.gz
rm 0012_asr_v1.tar.gz

# Parse i-vector extractor files  
cp ./0012_asr_v1/exp/nnet3_train_worn_simu_u400k_cleaned_rvb/extractor/* ./model/ivector_extractor/
rm ./model/ivector_extractor/final.ie.id

# Parse data files (words)
cp ./0012_asr_v1/data/lang/words.txt ./model/

# Parse HCLG
cp ./0012_asr_v1/exp/chain_train_worn_simu_u400k_cleaned_rvb/tree_sp/graph/HCLG.fst ./model/

# Parse main nn model (.mdl)
cp ./0012_asr_v1/exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/final.mdl ./model/

# Parse global cmvn stats
cp ./0012_asr_v1/exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/cmvn_opts ./model/
mv ./model/cmvn_opts ./model/global_cmvn.stats

# Clean unnecessary files
rm -r ./0012_asr_v1