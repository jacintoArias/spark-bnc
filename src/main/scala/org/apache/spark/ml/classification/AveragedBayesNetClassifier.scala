package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl._

import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._

class AveragedBayesNetClassificationModel private[ml] (
    override val uid: String,
    override val models: Array[BayesNetClassificationModel],
    override val numClasses: Int) 
  extends ProbabilisticClassificationModel[Vector, AveragedBayesNetClassificationModel]
with AveragedBayesNetModel[BayesNetClassificationModel] with BayesNetClassifierParams with Serializable {

  private[ml] def this(
    models: Array[BayesNetClassificationModel],
    numClasses: Int) =
    this(Identifiable.randomUID("bncensemble"), models, numClasses)


  protected def predictRaw(features: Vector): Vector = {
      val posterior = getClassPosterior(features)
      Vectors.dense(posterior)
  }
  
  protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in AveragedBayesNetrClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }


  // def transformBySubmodels(dataset: Dataset[_]): DataFrame =  {

  //   val predictUDF = udf { (features: Vector) =>

  //     val modelsPosteriors = getModelsClassPosterior(features)

  //     val ensemblePosteriors: Array[Double] = Array.range(0, numClasses).map(c => {
  //       modelsPosteriors.map(p => p(c)).sum
  //     })

  //     val modelsPredictions = modelsPosteriors.map(v => v.indexOf(v.max))
  //     val ensemblePredictions = ensemblePosteriors.indexOf(ensemblePosteriors.max)

  //     modelsPredictions.zipWithIndex :+ (ensemblePredictions, -1)
  //   }

  //   dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  // }

  override def copy(extra: ParamMap): AveragedBayesNetClassificationModel = defaultCopy(extra)
}