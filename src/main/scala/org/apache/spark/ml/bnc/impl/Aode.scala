package org.apache.spark.ml.bnc.impl

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.classification.{AveragedBayesNetClassificationModel, BayesNetClassificationModel}
import org.apache.spark.ml.bnc._

private[ml] object Aode {

  def run(
    input: RDD[LabeledPoint],
    numClasses: Int,
    categoricalFeatures: Map[Int, Int],
    parentUID: Option[String] = None): AveragedBayesNetClassificationModel = {

      // Build dataset metadata
      val metadata = BayesNetClassifierMetadata.buildMetadata(input, categoricalFeatures, numClasses)

      // Compute joint frequency tables
      val freqs: RDD[(Vector[Int], FreqTable)] = 
        FreqTable.computeAttributePairwiseFreqs(input, metadata)
            .cache

      // Compute children node cpts
      val childrenTables: Map[Vector[Int], Array[BncNode]] = 
        Aode.getChildrenTables(freqs, metadata)
            .groupBy{ case BncNode(att, parents, table) => parents}

      // Compute super-parent node cpts
      val parentsTables: Map[Vector[Int], Array[BncNode]] = 
        Aode.getParentsTables(freqs, metadata)
            .groupBy{ case BncNode(att, parents, table) => Vector(att)}

      // Compute class priors
      val classTable: ProbTable = getClassTable(freqs, metadata)

      // Group the nodes by spode
      val spodes: Array[BayesNetClassificationModel] = 
        Array.range(0, metadata.numFeatures).map(sp => {
          val key = Vector(sp)
          val nodes = parentsTables(key) ++ childrenTables(key) 
          new BayesNetClassificationModel(classTable, nodes, metadata.numClasses)
        })

      // Create the ensemble
      return new AveragedBayesNetClassificationModel(spodes, metadata.numClasses) 
    }

  // Children tables are easily obtained for each pair of attrbiutes X,Y:
  // P(X|Y,C)
  // P(Y|X,C)
  def getChildrenTables(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[BncNode] = {

    freqs.flatMap{ case (Vector(nodei, nodej), table) => {
        Vector(
          BncNode(nodej, Vector(nodei), table.conditionalProb(Vector(1), Vector(2, 0))),
          BncNode(nodei, Vector(nodej), table.conditionalProb(Vector(0), Vector(2, 1)))
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
          BncNode(sp, Vector(), table.conditionalProb(Vector(atts.indexOf(sp)), Vector(2))) 
        case None => 
          throw new Exception(f"Invalid Frequencies model for AODE. Cannot find any tuple ${sp}")
      }
    }
  }

  // Class priors can be marginalized from any freq table
  def getClassTable(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): ProbTable = {

    freqs.first._2.conditionalProb(Vector(), Vector(2))
  }


  // def sumModels(models : RDD[(String, Array[Int])]*) = {
  //   models.reduce(_ union _).reduceByKey((x,y) => (x,y).zipped.map(_+_))
  // }


  // def computeMi(model : RDD[(Vector[Int], FreqTable)]) = {
  //   model.flatMap{
  //     case (Vector(nodei, nodej), table) => this.genMI(table, nodei, nodej)
  //   }
  // }


  // def genMI(table : FreqTable,
  //           nodei : Int,
  //           nodej : Int
  //          ) : Vector[(Int, Double)] = {

  //   val mis = table.getMI()

  //   if (nodei != 0)
  //     return Vector()

  //   if (nodej == 1)
  //     Vector( (nodei, mis(1)), (nodej, mis(2)), (-1, 0.0))
  //   else
  //     Vector( (nodej, mis(2)) )
  // }
}