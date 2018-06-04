
package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl.KDependentBayes

class KDependentBayesClassifier (
    override val uid: String) 
  extends ProbabilisticClassifier[Vector, KDependentBayesClassifier, BayesNetClassificationModel] 
  with KDependentBayesClassifierParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("kdb"))

  override protected def train(dataset: Dataset[_]): BayesNetClassificationModel = {

    // Should these go in a nested structure, such as strategy in Dtrees?
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val numClasses: Int = getNumClasses(dataset)

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)

    val model = KDependentBayes.run(oldDataset, numClasses, categoricalFeatures, $(k))

    return model
  }

   override def setK(value: Int): this.type = set(k, value)

   override def copy(extra: ParamMap): KDependentBayesClassifier = defaultCopy(extra)

}


object KDependentBayesClassifier extends DefaultParamsReadable[KDependentBayesClassifier] {
  override def load(path: String): KDependentBayesClassifier = super.load(path)
}
