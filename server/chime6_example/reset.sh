#!/bin/sh

# Remove all (only)files in parent folder
find ./model -maxdepth 1 -type f -exec rm -rf '{}' \;

# Remove all files from i-vector extractor folder except 'online_cmvn_iextractor'
find ./model/ivector_extractor/ ! -name 'online_cmvn_iextractor' -type f -exec rm -f {} +