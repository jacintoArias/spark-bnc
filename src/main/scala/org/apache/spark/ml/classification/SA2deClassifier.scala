
package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl.SA2de


class SA2deClassifier (
    override val uid: String) 
  extends ProbabilisticClassifier[Vector, SA2deClassifier, AveragedBayesNetClassificationModel] 
  with BayesNetClassifierParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("sa2de"))

  override protected def train(dataset: Dataset[_]): AveragedBayesNetClassificationModel = {

    // Should these go in a nested structure, such as strategy in Dtrees?
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val numClasses: Int = getNumClasses(dataset)

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)

    val model = SA2de.run(oldDataset, numClasses, categoricalFeatures)

    return model
  }

   override def copy(extra: ParamMap): SA2deClassifier = defaultCopy(extra)

}


object SA2deClassifier extends DefaultParamsReadable[SA2deClassifier] {
  override def load(path: String): SA2deClassifier = super.load(path)
}