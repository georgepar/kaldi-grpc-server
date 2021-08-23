#!/usr/bin/env bash

MODEL_DIR=$1
TAG=$2

if [ -z "$(ls model)"  ]; then
    cp -r $MODEL_DIR/* model/
    docker build -t $TAG -f Dockerfile .
else
    echo "Expecting ./model directory to be empty"
    exit 1
fi
