
package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl.A2de


class A2deClassifier (
    override val uid: String) 
  extends ProbabilisticClassifier[Vector, A2deClassifier, AveragedBayesNetClassificationModel] 
  with BayesNetClassifierParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("aode"))

  override protected def train(dataset: Dataset[_]): AveragedBayesNetClassificationModel = {

    // Should these go in a nested structure, such as strategy in Dtrees?
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val numClasses: Int = getNumClasses(dataset)

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)

    val model = A2de.run(oldDataset, numClasses, categoricalFeatures)

    return model
  }

   override def copy(extra: ParamMap): A2deClassifier = defaultCopy(extra)

}


object A2deClassifier extends DefaultParamsReadable[A2deClassifier] {
  override def load(path: String): A2deClassifier = super.load(path)
}