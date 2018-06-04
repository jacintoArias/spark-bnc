#!/bin/bash

# Build in dockerized sbt 
#

docker run \
    --rm \
    -v $(pwd):/home/work/project \
    jacintoarias/docker-sparkdev \
    sbt "examples / package" \
    -ivy ./.ivy
