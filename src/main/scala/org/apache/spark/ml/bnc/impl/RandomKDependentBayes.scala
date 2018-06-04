package org.apache.spark.ml.bnc.impl

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.classification.{BayesNetClassificationModel, AveragedBayesNetClassificationModel}
import org.apache.spark.ml.bnc._

import scala.util.Random

private[ml] object RandomKDependentBayes {

  def run(
    input: RDD[LabeledPoint],
    numClasses: Int,
    categoricalFeatures: Map[Int, Int],
    k: Int,
    numModels: Int,
    subsamplingRate: Double,
    seed: Long): AveragedBayesNetClassificationModel = {

      val rnd = new Random(seed)

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

      // Get randomized best parents for each model
      val parents: Array[Array[(Int, Vector[Int])]] =
        Array.range(0, numModels).map { model =>
          for (att <- Array.range(0, metadata.numFeatures)) yield {

            val candidates =  sigma
                // Get the atts in preceding order
                .slice(0, sigma.indexOf(att))
                // Zip with the score
                .map( parent => (parent, condiMi(att)(parent)) ) 
                // order
                .sortBy{ case (parent, score) => -score }

            val n = candidates.length 
            val nSampled = math.max(candidates.length * subsamplingRate, k).toInt
            val idSampled = rnd.shuffle( Vector.range(0, n) ).slice(0, nSampled).sorted

            val candidatesSampled = 
              idSampled.map(x => candidates(x))
              // Get k best
              .slice(0, k)
              // Map back to index
              .map{ case (parent, score) => parent }
              .sorted

            (att, candidatesSampled)
        } } 
        
      // Aggregate all parentsets in a single structure
      val allParents: Array[Vector[Int]] = parents.flatMap{ v => v.map{ case (i, par) => (i +: par) } }.toSet.toVector.toArray 

      // Compute multidimensional joint frequency tables
      val freqsMulti: RDD[(Vector[Int], FreqTable)] = FreqTable.computeFreqsGeneric(input, metadata, allParents).cache

      // Compute children node cpts
      val attributeNodes: Map[Vector[Int], BncNode] = RandomKDependentBayes.getAttributeTables(freqsMulti, metadata)

      val modelsNodes: Array[Array[BncNode]] = parents.map(v => 
        v.map { case (i, par) =>
          val key = (i +: par).toVector
          attributeNodes(key)
        }
      )

      // println("KDB")
      // attributeNodes.foreach { case BncNode(att, parents, table) => {

      //     println(f"Att: $att")
      //     println(f"""Parents: ${parents.mkString(",")}""")
      //     println(f"""Table Arity: ${table.featureArity}""")
      //     // println(f"""Table: ${table.params.mkString(",")}""")
      // } }

      // Compute class priors
      val classTable: ProbTable = getClassTable(freqs, metadata)

      val models = modelsNodes.map( ms => new BayesNetClassificationModel(classTable, ms, metadata.numClasses) )

      new AveragedBayesNetClassificationModel(models, metadata.numClasses) 
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
    metadata: BayesNetClassifierMetadata): Map[Vector[Int], BncNode] = {

    freqs.map{ case (atts, table) =>{
      val att = atts(0)
      val par = atts.tail
      val Z = Vector.range(1, par.length+2).reverse // class (last element) +: parents
      (atts, BncNode(att, par, table.conditionalProb(Vector(0), Z)))
    } }
    .collect
    .toMap
  }


  // Class priors can be marginalized from any freq table
  def getClassTable(
    freqs: RDD[(Vector[Int], FreqTable)],
    metadata: BayesNetClassifierMetadata): ProbTable = {

    val table: FreqTable = freqs.first._2
    table.conditionalProb(Vector(), Vector(table.featureArity.length-1))
  }
}