# spark-bnc package

This package implements discrete Bayesian network classifiers following the same API than most of Spark MLlib classifiers.

The current classifiers are implemented:

* Discrete Naive Bayes
* Tree Augmented Naive Bayes (TAN)
* K-Dependent Bayesian Classifier (KDB)
* Averaged one-Dependent Estimators (AODE, A1DE)
* Averaged two-Dependent Estimators (A2DE)

It implements serveral utilities to extend these implementation to other discrete Bayesian network models by implementing graph structures, conditional probability distributions and Laplace Smoothing.

## Usage

The library builds on top of Apache Spark private modules to extend the internal capabilities of MLlib. You can find the modules by importing:

```scala
import org.apache.spark.ml.classification.{DiscreteNaiveBayesClassifier, AodeClassifier}

val nb = new DiscreteNaiveBayesClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)

val aode = new AodeClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)
```

A comprehensive example is located in https://github.com/jacintoArias/spark-bnc/blob/master/examples/src/main/scala/es/jarias/spark/examples/ml/RunAll.scala

## Developing

Code is well documented and is self explanatory. Feel free to extend it for your own research, I will be pleased to answer any questions about the structure and give further advice in how to develop additional solutions. Thank you for your interest in my job!

I provide a dockerized build and run examples using sbt in `scripts`. If you have docker installed it should build and run into a local spark deploy by just running the scripts.

## Motivation

This work is a compilation of algorithms and utilities created during my PhD research work. Thesis is still in review, I will publish a link to it as soon as it is available.
