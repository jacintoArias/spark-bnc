package org.apache.spark.ml.bnc

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.bnc.impl._


private[spark] trait BayesNetModel {

    def classPriors: ProbTable 

    def numClasses: Int

    def nodes: Array[BncNode]

    def nParams: Int = nodes.map(n => n.nParams).sum + classPriors.nParams

    def getClassPosterior(features: Vector): Array[Double] = {

       val feat = features.toArray.map(_.toInt)

       val probs =
        try {
            Array.range(0, numClasses).map(c => {
            classPriors.value(c) * 
                nodes.map{ case BncNode(att, parents, table) => {
                    val idx: Seq[Int] = c +: parents.map(feat(_).toInt) :+ feat(att).toInt
                    table.value(idx: _*)
                } }.product
            })
        } catch {
            case e: Exception => Array.fill(numClasses)(1.toDouble/numClasses)
        }

        return probs
    }
}

private[spark] trait AveragedBayesNetModel[M <: BayesNetModel] {

    def numClasses: Int

    def models: Array[M]

    def nParams: Int = models.map(m => m.nParams).sum


    def getClassPosterior(features: Vector): Array[Double] = {

        val spodeProbs: Array[Array[Double]] = getModelsClassPosterior(features)

        val probs: Array[Double] = Array.range(0, numClasses).map(c => {
            spodeProbs.map(p => p(c)).sum / spodeProbs.length
        })

        probs
    }

    def getModelsClassPosterior(features: Vector): Array[Array[Double]] = {

        val spodeProbs: Array[Array[Double]] = models.map(_.getClassPosterior(features))

       spodeProbs 
    }
}