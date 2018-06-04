#!/bin/bash

# Build in dockerized sbt 
#

docker run \
    --rm \
    -v $(pwd):/home/work/project \
    -v $(pwd)/.ivy:/sbtlib \
    jacintoarias/docker-sparkdev \
    sbt package
