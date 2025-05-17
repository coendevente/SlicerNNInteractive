#!/bin/bash

docker run --rm \
    --volume $PWD/paper:/data \
    --user $(id -u):$(id -g) \
    openjournals/inara \
    -o pdf,preprint \
    paper.md
