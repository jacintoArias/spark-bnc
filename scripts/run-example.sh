#!/bin/bash

# Run in dockerized spark
# $1 is the main example class to run
# Remaining arguments are taken in orde rby the app
#

docker run \
    --rm \
    -v $(pwd):/home/work/project \
    -v $(pwd)/.ivy:/sbtlib \
    jacintoarias/docker-sparkdev \
    spark-submit \
    --class es.jarias.spark.examples.ml.bnc.$1 \
    --jars target/scala-2.11/spark-bnc_2.11-0.1.0.jar \
    examples/target/scala-2.11/spark-bnc-examples_*.jar \
    ${@:2}
