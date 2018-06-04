package org.apache.spark.ml.bnc.impl

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.classification.{AveragedBayesNetClassificationModel, BayesNetClassificationModel}
import org.apache.spark.ml.bnc._

private[ml] object SA2de {

  def run(
    input: RDD[LabeledPoint],
    numClasses: Int,
    categoricalFeatures: Map[Int, Int],
    parentUID: Option[String] = None): AveragedBayesNetClassificationModel = {

      // Build dataset metadata
      val metadata = BayesNetClassifierMetadata.buildMetadata(input, categoricalFeatures, numClasses)

      // Compute joint frequency tables
      val freqs2X: RDD[(Vector[Int], FreqTable)] = FreqTable.computeAttributePairwiseFreqs(input, metadata).cache

      val selected: Array[(Int, Int)] = 
        getPairwiseMutualInformation(freqs2X, metadata) 
          .sortBy { case (i, j, mi) => -mi }
          .map { case (i, j, mi) => (i, j) }
          .slice(0, metadata.numFeatures * 2)

      // Compute joint frequency tables
      val freqs: RDD[(Vector[Int], FreqTable)] = 
        FreqTable.computeAttributeX3FreqsCaped(input, selected, metadata)
            .cache

      // Compute children node cpts
      val childrenTables: Map[Vector[Int], Array[BncNode]] = 
        SA2de.getChildrenTables(freqs, metadata)
            .groupBy{ case BncNode(att, parents, table) => parents}

      // Compute super-parent node cpts
      val parentsTables: Map[Int, Array[BncNode]] = 
        SA2de.getParentsTables(freqs, selected, metadata)
            .groupBy{ case BncNode(att, parents, table) => att}

      // Compute class priors
      val classTable: ProbTable = getClassTable(freqs, metadata)

    //   println(f""" Selected: ${selected.mkString(",")}""")
    //   println(f""" ChildrenTables:""")
    //  childrenTables.foreach { case (att, nodes) => {
    //    println(f"Parents: $att")
    //    nodes.foreach { case BncNode(child, parents, table) => println(f""" $child """) }
    //  } } 

      // Group the nodes by spode
      val spodes: Array[BayesNetClassificationModel] = 
        selected.map { case (spi, spj) => {
          val key = Vector(spi, spj)
          val nodes = parentsTables(spi) ++ parentsTables(spj) ++ childrenTables(key) 
          new BayesNetClassificationModel(classTable, nodes, metadata.numClasses)
        } }

      // Create the ensemble
      return new AveragedBayesNetClassificationModel(spodes, metadata.numClasses) 
    }


  // Attribute tables are easily obtained for each pair of attribute X:
  // P(X,C)
  def getPairwiseMutualInformation(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[(Int, Int, Double)] = {

    freqs.map{ case (Vector(nodei, nodej), table) => {
      val mi = table.getJointMutualInformation() 
      (nodei, nodej, mi)
    } }
    .collect
  }

  // Children tables are easily obtained for each pair of attrbiutes X,Y:
  // P(X|Y,C)
  // P(Y|X,C)
  def getChildrenTables(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[BncNode] = {

    freqs.map{ case (Vector(nodei, nodej, nodek), table) => {
          // println(f"$nodei, $nodej, $nodek")
          BncNode(nodek, Vector(nodei, nodej), table.conditionalProb(Vector(2), Vector(3, 0, 1)))
    } }
    .collect
  }

 // Parent tables are obtained from 0-index frequency tables that should contain all possible attributes
 def getParentsTables(
    freqs: RDD[(Vector[Int], FreqTable)],
    pairs: Array[(Int, Int)],
    metadata: BayesNetClassifierMetadata): Array[BncNode] = {

    val base = freqs.collect()

    pairs.flatMap { case (i, j) => {
      
      val table = base.find{ case (atts, probs) => atts.contains(i) && atts.contains(j) }

      table match {
        case Some((atts, table)) => 
          // Find the sp index in the freq table and compute P(SP|C)
          Vector(
            BncNode(i, Vector(), table.conditionalProb(Vector(atts.indexOf(i)), Vector(3))),
            BncNode(j, Vector(), table.conditionalProb(Vector(atts.indexOf(j)), Vector(3)))  
          )
        case None => 
          throw new Exception(f"Invalid Frequencies model for AODE. Cannot find any tuple ${i} ${j}")
      } }
    }
  }


  // Class priors can be marginalized from any freq table
  def getClassTable(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): ProbTable = {

    freqs.first._2.conditionalProb(Vector(), Vector(3))
  }
}