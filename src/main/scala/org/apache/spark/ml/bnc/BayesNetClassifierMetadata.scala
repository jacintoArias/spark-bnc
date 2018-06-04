package org.apache.spark.ml.bnc

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD

class BayesNetClassifierMetadata(
  val numFeatures: Int,
  val numExamples: Long,
  val numClasses: Int,
  val featureArity: Vector[Int]) extends Serializable {

}

object BayesNetClassifierMetadata {

  def buildMetadata(
    input: RDD[LabeledPoint],
    categoricalFeaturesInfo: Map[Int, Int],
    labelInfo: Int): BayesNetClassifierMetadata = {

      val numFeatures = input.map(_.features.size).take(1).headOption.getOrElse {
        throw new IllegalArgumentException(s"BayesClassifier requires size of input RDD > 0, " +
          s"but was given by empty one.")
      }

      require(numFeatures > 0, s"BayesClassifier requires number of features > 0, " +
        s"but was given an empty features vector")

      Vector.range(0, numFeatures).foreach( f =>
        require(categoricalFeaturesInfo.toVector.length == numFeatures, f"All feature should be categorical with positive number of states, " +
          s"categoricalFeaturesInfo for feature $f is < 0")
      );

      val numExamples = input.count()
      val numClasses = labelInfo
      val featureArity = Vector.range(0, numFeatures).map(categoricalFeaturesInfo(_))
      // val parentSet = for (_ <- Array.range(0, numFeatures)) yield Set[Int]()

      new BayesNetClassifierMetadata(numFeatures, numExamples, numClasses, featureArity)
    }
  }
