package org.apache.spark.ml.bnc.impl

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.classification.BayesNetClassificationModel
import org.apache.spark.ml.bnc._

private[ml] object KDependentBayes {

  def run(
    input: RDD[LabeledPoint],
    numClasses: Int,
    categoricalFeatures: Map[Int, Int],
    k: Int,
    parentUID: Option[String] = None): BayesNetClassificationModel = {

      // Build dataset metadata
      val metadata = BayesNetClassifierMetadata.buildMetadata(input, categoricalFeatures, numClasses)

      // Compute joint frequency tables
      val freqs: RDD[(Vector[Int], FreqTable)] = FreqTable.computeAttributePairwiseFreqs(input, metadata).cache

      // Get mutual information
      val mi: Array[(Int, Double)] = getMutualInformation(freqs, metadata) 

      val condiMi: Map[Int, Map[Int, Double]] = 
        getPairwiseMutualInformation(freqs, metadata) 
          // Group by i
          .groupBy{ case (att, candidate, mi) => att }
          .mapValues{ case col => 
            // Group internal by j
            col
              .map{ case (att, candidate, mi) => (candidate, mi) }
              .toMap 
          }

      // Get variable ordering
      val sigma = mi.sortBy{ case (att, mi) => -mi }.map(_._1)

      // Get best parents
      val parents: Array[Array[Int]] = 
        for (att <- Array.range(0, metadata.numFeatures)) yield
          sigma
            // Get the atts in preceding order
            .slice(0, sigma.indexOf(att))
            // Zip with the score
            .map( parent => (parent, condiMi(att)(parent)) ) 
            // order
            .sortBy{ case (parent, score) => -score }
            // Get k best
            .slice(0, k)
            // Map back to index
            .map{ case (parent, score) => parent }
            .sorted

      // Compute multidimensional joint frequency tables
      val freqsMulti: RDD[(Vector[Int], FreqTable)] = FreqTable.computeFreqsParentSet(input, metadata, parents).cache

      // Compute children node cpts
      val attributeNodes: Array[BncNode] = KDependentBayes.getAttributeTables(freqsMulti, parents, metadata)

      // println("KDB")
      // mi.sortBy(_._1).foreach( t => println(t))
      // println(f"""Sigma: ${sigma.mkString(",")}""")
      // condiMi.foreach { case (att, scores) => { 
      //   println(f"$att")
      //   scores.toVector.sortBy(t => -t._2).foreach { case (par, sco) => println(f"\t$par $sco") }
      // } }
      // attributeNodes.foreach { case BncNode(att, parents, table) => {

      //     println(f"Att: $att")
      //     println(f"""Parents: ${parents.mkString(",")}""")
      //     println(f"""Table Arity: ${table.featureArity}""")
      //     // println(f"""Table: ${table.params.mkString(",")}""")
      // } }

      // Compute class priors
      val classTable: ProbTable = getClassTable(freqs, metadata)

      return new BayesNetClassificationModel(classTable, attributeNodes, metadata.numClasses)
    }


  // Attribute tables are easily obtained for each pair of attribute X:
  // P(X,C)
  def getPairwiseMutualInformation(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[(Int, Int, Double)] = {

    freqs.flatMap{ case (Vector(nodei, nodej), table) => {
      val mi = table.getJointMutualInformation() 
      Vector(
        (nodei, nodej, mi),
        (nodej, nodei, mi)
      )
    } }
    .collect
  }

  // Parent tables are obtained from 0-index frequency tables that should contain all possible attributes
  def getMutualInformation(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): Array[(Int, Double)] = {

    val base = freqs.filter{ case (atts, table) => atts.contains(0) }.collect()

    for (att <- Array.range(0, metadata.numFeatures))
    yield {
      val table = base.find{ case (atts, probs) => atts.contains(att) }
      table match {
        case Some((atts, table)) => 
          // Find the sp index in the freq table and compute P(SP|C)
          (att, table.getMutualInformation(atts.indexOf(att), Vector(2))) 
        case None => 
          throw new Exception(f"Invalid Frequencies model for KDependentBayes. Cannot find any tuple ${att}")
      }
    }
  }

  // Attribute tables are easily obtained for each attrbiute X and its parents:
  // P(X,C)
  def getAttributeTables(
    freqs: RDD[(Vector[Int], FreqTable)],
    parents: Array[Array[Int]],
    metadata: BayesNetClassifierMetadata): Array[BncNode] = {

    freqs.map{ case (Vector(nodei), table) => {
      val par = parents(nodei).toVector
      val Z = Vector.range(1, par.length+2).reverse // class (last element) :+ parents
      // Add the class to the conditionals (must be last variable in the table)
      BncNode(nodei, par, table.conditionalProb(Vector(0), Z))
    } }
    .collect
  }


  // Class priors can be marginalized from any freq table
  def getClassTable(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): ProbTable = {

    val table: FreqTable = freqs.first._2
    table.conditionalProb(Vector(), Vector(table.featureArity.length-1))
  }
}