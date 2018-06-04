
package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasSeed}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl.RandomKDependentBayes

class RandomKDependentBayesClassifier (
    override val uid: String) 
  extends ProbabilisticClassifier[Vector, RandomKDependentBayesClassifier, AveragedBayesNetClassificationModel] 
  with RandomKDependentBayesClassifierParams with HasSeed with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("rkdb"))

  override protected def train(dataset: Dataset[_]): AveragedBayesNetClassificationModel = {

    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val numClasses: Int = getNumClasses(dataset)

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)

    val model = RandomKDependentBayes.run(oldDataset, numClasses, categoricalFeatures, $(k), $(numModels), $(subsamplingRate), $(seed))

    return model
  }

  override def setK(value: Int): this.type = set(k, value)

  override def setNumModels(value: Int): this.type = set(numModels, value)

  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

   override def copy(extra: ParamMap): RandomKDependentBayesClassifier = defaultCopy(extra)

}


object RandomKDependentBayesClassifierParams extends DefaultParamsReadable[RandomKDependentBayesClassifierParams] {
  override def load(path: String): RandomKDependentBayesClassifierParams = super.load(path)
}
