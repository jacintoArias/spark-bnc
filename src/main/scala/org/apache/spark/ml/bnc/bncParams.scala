package org.apache.spark.ml.bnc

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._


private[ml] trait BayesNetClassifierParams extends PredictorParams {

}

private[ml] trait KDependentBayesClassifierParams 
extends BayesNetClassifierParams {

  final val k: IntParam = new IntParam(this, "k", "Maximum parents allowed",
    ParamValidators.gtEq(1))

  setDefault(k -> 1)

  def setK(value: Int): this.type = set(k, value)

  /** @group getParam */
  final def getK: Int = $(k)
}

private[ml] trait RandomKDependentBayesClassifierParams 
extends KDependentBayesClassifierParams {

  final val numModels: IntParam = new IntParam(this, "numModels", "Number of independent models",
    ParamValidators.gtEq(1))

  final val subsamplingRate: DoubleParam = new DoubleParam(this, "subsamplingRate", "Proportion of candidates to consider",
    ParamValidators.inRange(0, 1))

  setDefault(k -> 10)
  setDefault(subsamplingRate -> 0.8)

  def setNumModels(value: Int): this.type = set(numModels, value)

  def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group getParam */
  final def getNumModels: Int = $(numModels)
  
  final def getSubsamplingRate: Double = $(subsamplingRate)

}