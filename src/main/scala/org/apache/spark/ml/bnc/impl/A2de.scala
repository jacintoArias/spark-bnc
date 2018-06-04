package org.apache.spark.ml.bnc.impl

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.classification.{AveragedBayesNetClassificationModel, BayesNetClassificationModel}
import org.apache.spark.ml.bnc._

private[ml] object A2de {

  def run(
    input: RDD[LabeledPoint],
    numClasses: Int,
    categoricalFeatures: Map[Int, Int],
    parentUID: Option[String] = None): AveragedBayesNetClassificationModel = {

      // Build dataset metadata
      val metadata = BayesNetClassifierMetadata.buildMetadata(input, categoricalFeatures, numClasses)

      // Compute joint frequency tables
      val freqs: RDD[(Vector[Int], FreqTable)] = 
        FreqTable.computeAttributeX3Freqs(input, metadata)
            .cache

      // Compute children node cpts
      val childrenTables: Map[Vector[Int], Array[BncNode]] = 
        A2de.getChildrenTables(freqs, metadata)
            .groupBy{ case BncNode(att, parents, table) => parents}

      // Compute super-parent node cpts
      val parentsTables: Map[Int, Array[BncNode]] = 
        A2de.getParentsTables(freqs, metadata)
            .groupBy{ case BncNode(att, parents, table) => att}

      // Compute class priors
      val classTable: ProbTable = getClassTable(freqs, metadata)

      // Group the nodes by spode
      val spodes: Array[BayesNetClassificationModel] = 
        for (
          spi <- Array.range(0, metadata.numFeatures);
          spj <- Array.range(spi+1, metadata.numFeatures)
        ) yield {
          val key = Vector(spi, spj)
          val nodes = parentsTables(spi) ++ parentsTables(spj) ++ childrenTables(key) 
          new BayesNetClassificationModel(classTable, nodes, metadata.numClasses)
        }

      // Create the ensemble
      return new AveragedBayesNetClassificationModel(spodes, metadata.numClasses) 
    }

  // Children tables are easily obtained for each pair of attrbiutes X,Y:
  // P(X|Y,C)
  // P(Y|X,C)
  def getChildrenTables(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[BncNode] = {

    freqs.flatMap{ case (Vector(nodei, nodej, nodek), table) => {
        Vector(
          BncNode(nodei, Vector(nodej, nodek), table.conditionalProb(Vector(0), Vector(3, 1, 2))),
          BncNode(nodej, Vector(nodei, nodek), table.conditionalProb(Vector(1), Vector(3, 0, 2))),
          BncNode(nodek, Vector(nodei, nodej), table.conditionalProb(Vector(2), Vector(3, 0, 1)))
        )
    } }
    .collect
  }

 // Parent tables are obtained from 0-index frequency tables that should contain all possible attributes
 def getParentsTables(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[BncNode] = {

    val base = freqs.filter{ case (atts, table) => atts.contains(0) }.collect()

    for (sp <- Array.range(0, metadata.numFeatures))
    yield {
      val table = base.find{ case (atts, probs) => atts.contains(sp) }
      table match {
        case Some((atts, table)) => 
          // Find the sp index in the freq table and compute P(SP|C)
          BncNode(sp, Vector(), table.conditionalProb(Vector(atts.indexOf(sp)), Vector(3))) 
        case None => 
          throw new Exception(f"Invalid Frequencies model for AODE. Cannot find any tuple ${sp}")
      }
    }
  }

  // Class priors can be marginalized from any freq table
  def getClassTable(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): ProbTable = {

    freqs.first._2.conditionalProb(Vector(), Vector(3))
  }
}