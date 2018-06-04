package org.apache.spark.ml.bnc.impl

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.classification.BayesNetClassificationModel
import org.apache.spark.ml.bnc._

private[ml] object NaiveBayes {

  def run(
    input: RDD[LabeledPoint],
    numClasses: Int,
    categoricalFeatures: Map[Int, Int],
    parentUID: Option[String] = None): BayesNetClassificationModel = {

      // Build dataset metadata
      val metadata = BayesNetClassifierMetadata.buildMetadata(input, categoricalFeatures, numClasses)

      // Compute joint frequency tables
      val freqs: RDD[(Vector[Int], FreqTable)] = FreqTable.computeAttributeFreqs(input, metadata).cache

      // Compute children node cpts
      val attributeNodes: Array[BncNode] = NaiveBayes.getAttributeTables(freqs, metadata)

      // Compute class priors
      val classTable: ProbTable = getClassTable(freqs, metadata)

      return new BayesNetClassificationModel(classTable, attributeNodes, metadata.numClasses)
    }


  // Attribute tables are easily obtained for each pair of attrbiute X:
  // P(X,C)
  def getAttributeTables(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[BncNode] = {

    freqs.map{ case (Vector(nodei), table) =>
          BncNode(nodei, Vector(), table.conditionalProb(Vector(0), Vector(1)))
    }
    .collect
  }


  // Class priors can be marginalized from any freq table
  def getClassTable(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): ProbTable = {

    freqs.first._2.conditionalProb(Vector(), Vector(1))
  }
}