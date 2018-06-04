package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl._

class BayesNetClassificationModel private[ml] (
    override val uid: String,
    override val classPriors: ProbTable,
    override val nodes: Array[BncNode],
    override val numClasses: Int) 
  extends ProbabilisticClassificationModel[Vector, BayesNetClassificationModel]
with BayesNetModel with BayesNetClassifierParams with Serializable {

    private[ml] def this(
        classPriors: ProbTable,
        nodes: Array[BncNode],
        numClasses: Int) =
        this(Identifiable.randomUID("bnc"), classPriors, nodes, numClasses)


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
                throw new RuntimeException("Unexpected error in BayesNetClassificationModel:" +
                " raw2probabilityInPlace encountered SparseVector")
        }
    }
  

    override def copy(extra: ParamMap): BayesNetClassificationModel = defaultCopy(extra)
}