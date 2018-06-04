
package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.util._

import org.apache.spark.ml.bnc._
import org.apache.spark.ml.bnc.impl.Aode


class AodeClassifier (
    override val uid: String) 
  extends ProbabilisticClassifier[Vector, AodeClassifier, AveragedBayesNetClassificationModel] 
  with BayesNetClassifierParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("aode"))

  override protected def train(dataset: Dataset[_]): AveragedBayesNetClassificationModel = {

    // Should these go in a nested structure, such as strategy in Dtrees?
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val numClasses: Int = getNumClasses(dataset)

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)

    val model = Aode.run(oldDataset, numClasses, categoricalFeatures)

    return model
  }

   override def copy(extra: ParamMap): AodeClassifier = defaultCopy(extra)

}

object AodeClassifier extends DefaultParamsReadable[AodeClassifier] {
  override def load(path: String): AodeClassifier = super.load(path)
}