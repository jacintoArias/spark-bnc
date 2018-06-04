package org.apache.spark.ml.bnc.impl

import org.apache.spark.rdd.RDD

class ProbTable private[ml] (
    val params: Vector[Double],
    val featureArity: Vector[Int]) 
extends Serializable {

  def getOffsets(featureArity: Vector[Int]): Vector[Int] = featureArity.scanRight(1)(_ * _).tail

  val offsets = getOffsets(featureArity)

  def idx(conf: Seq[Int], offsets: Vector[Int]): Int = conf.zip(offsets).map(t => t._1 * t._2).sum

  def value(conf: Int*): Double = params(idx(conf, offsets))

  def nParams(): Int = params.length
}