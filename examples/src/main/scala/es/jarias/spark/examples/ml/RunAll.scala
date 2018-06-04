package es.jarias.spark.examples.ml.bnc

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.util.SizeEstimator

import org.apache.spark.ml.bnc._

import org.apache.log4j.{Level, Logger, LogManager, PropertyConfigurator}

object RunAll {

  def main(args: Array[String]) {

    val MB = 1024 * 1024

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val log = LogManager.getLogger("Experiment")
    log.setLevel(Level.DEBUG)
    log.debug("Start")

    val path = "data/iris.csv"

    val spark = SparkSession
      .builder()
      .appName(f"Experiment $path")
      .config("spark.driver.maxResultSize", "1024")
      .getOrCreate()


    val df = spark
      .read
      .format("csv")
      .option("header", "false")
      .option("inferSchema", "true")
      .load(f"$path")

    val featuresDiscrete = df.schema.toArray.init
      .filter { case StructField(name, fieldType, _, _) => (fieldType != DoubleType) && (fieldType != FloatType) }
      .map { case StructField(name, fieldType, _, _) => name }

    val featuresContinuous = df.schema.toArray.init
      .filter { case StructField(name, fieldType, _, _) => (fieldType == DoubleType) || (fieldType == FloatType) }
      .map { case StructField(name, fieldType, _, _) => name }
    
    val featuresDiscretized = featuresContinuous.map(feat => f"${feat}_disc")

    val label = df.columns.last

    val labelIndexer = new StringIndexer()
    .setInputCol(label)
    .setOutputCol("indexedLabel")

    val discretizer = new QuantileDiscretizer()
      .setInputCols(featuresContinuous)
      .setOutputCols(featuresDiscretized)
      .setNumBuckets(4)

    val assembler = new VectorAssembler()
      .setInputCols(featuresDiscrete ++ featuresDiscretized)
      .setOutputCol("features")

    val featureIndexer = new VectorIndexer()
      .setInputCol(assembler.getOutputCol)
      .setOutputCol("indexedFeatures")
      .setMaxCategories(15)

    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), 1234)

    val prePipeline = new Pipeline()
      .setStages(Array(labelIndexer, discretizer, assembler, featureIndexer)) 
      .fit(trainingData)

    val trainingDataPre = prePipeline.transform(trainingData).cache
    val testDataPre = prePipeline.transform(testData).cache

    log.debug("Training NB")
    val nb = new DiscreteNaiveBayesClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)
    log.debug(f"NB num params: ${nb.nParams}")
    log.debug(f"NB Estimated size: ${SizeEstimator.estimate(nb)/MB} MB")

    log.debug("Training KDB k=1")
    val kdb1 = new KDependentBayesClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setK(2)
      .fit(trainingDataPre)
    log.debug(f"KDB1 num params: ${kdb1.nParams}")
    log.debug(f"KDB1 Estimated size: ${SizeEstimator.estimate(kdb1)/MB} MB")

    log.debug("Training KDB k=2")
    val kdb2 = new KDependentBayesClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setK(2)
      .fit(trainingDataPre)
    log.debug(f"KDB2 num params: ${kdb2.nParams}")
    log.debug(f"KDB2 Estimated size: ${SizeEstimator.estimate(kdb2)/MB} MB")

    log.debug("Training KDB k=3")
    val kdb3 = new KDependentBayesClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setK(3)
      .fit(trainingDataPre)
    log.debug(f"KDB3 num params: ${kdb3.nParams}")
    log.debug(f"KDB3 Estimated size: ${SizeEstimator.estimate(kdb3)/MB} MB")

    log.debug("Training KDB k=4")
    val kdb4 = new KDependentBayesClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setK(4)
      .fit(trainingDataPre)
    log.debug(f"KDB4 num params: ${kdb4.nParams}")
    log.debug(f"KDB4 Estimated size: ${SizeEstimator.estimate(kdb4)/MB} MB")

    log.debug("Training RKDB k=1")
    val rkdb1 = new RandomKDependentBayesClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .setK(2)
      .setNumModels(10)
      .setSubsamplingRate(0.5)
      .fit(trainingDataPre)
    log.debug(f"RKDB1 num params: ${rkdb1.nParams}")
    log.debug(f"RKDB1 Estimated size: ${SizeEstimator.estimate(rkdb1)/MB} MB")

    log.debug("Training AODE")
    val aode = new AodeClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)
    log.debug(f"AODE num params: ${aode.nParams}")
    log.debug(f"AODE Estimated size: ${SizeEstimator.estimate(aode)/MB} MB")

    log.debug("Training A2DE")
    val a2de = new A2deClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)
    log.debug(f"A2DE num params: ${a2de.nParams}")
    log.debug(f"A2DE Estimated size: ${SizeEstimator.estimate(a2de)/MB} MB")

    log.debug("Training SA2DE")
    val sa2de = new SA2deClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)
    log.debug(f"SA2DE num params: ${sa2de.nParams}")
    log.debug(f"SA2DE Estimated size: ${SizeEstimator.estimate(sa2de)/MB} MB")

    log.debug("Training DecisionTree")
    val dt = new DecisionTreeClassifier()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)
    log.debug(f"DT Estimated size: ${SizeEstimator.estimate(dt)/MB} MB")

    log.debug("Training RandomForest")
    val rf = new RandomForestClassifier()
      .setNumTrees(50)
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(featureIndexer.getOutputCol)
      .fit(trainingDataPre)
    log.debug(f"RF Estimated size: ${SizeEstimator.estimate(rf)/MB} MB")


    log.debug("Evaluating")
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

      log.debug("Testing nb")
      val predictionsNb = nb.transform(testDataPre)
      println("NB = " + evaluator.evaluate(predictionsNb))

      log.debug("Testing kbd1")
      val predictionsKdb1 = kdb1.transform(testDataPre)
      println("KDB1 = " + evaluator.evaluate(predictionsKdb1))

      log.debug("Testing kdb2")
      val predictionsKdb2 = kdb2.transform(testDataPre)
      println("KDB2 = " + evaluator.evaluate(predictionsKdb2))

      log.debug("Testing kdb3")
      val predictionsKdb3 = kdb3.transform(testDataPre)
      println("KDB3 = " + evaluator.evaluate(predictionsKdb3))

      log.debug("Testing kdb4")
      val predictionsKdb4 = kdb4.transform(testDataPre)
      println("KDB4 = " + evaluator.evaluate(predictionsKdb4))

      log.debug("Testing rkdb1")
      val predictionsRkdb1 = rkdb1.transform(testDataPre)
      println("RKDB1 = " + evaluator.evaluate(predictionsRkdb1))

      log.debug("Testing aode")
      val predictionsAode = aode.transform(testDataPre)
      println("AODE = " + evaluator.evaluate(predictionsAode))

      log.debug("Testing a2de")
      val predictionsA2de = a2de.transform(testDataPre)
      println("A2DE = " + evaluator.evaluate(predictionsA2de))

      log.debug("Testing sa2de")
      val predictionsSA2de = sa2de.transform(testDataPre)
      println("SA2DE = " + evaluator.evaluate(predictionsSA2de))

      log.debug("Testing dt")
      val predictionsDt = dt.transform(testDataPre)
      println("DT = " + evaluator.evaluate(predictionsDt))

      log.debug("Testing rf")
      val predictionsRf = rf.transform(testDataPre)
      println("RF = " + evaluator.evaluate(predictionsRf))
  }
}