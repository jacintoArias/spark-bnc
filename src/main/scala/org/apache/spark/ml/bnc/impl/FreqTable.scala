package org.apache.spark.ml.bnc.impl

import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.LabeledPoint

import org.apache.spark.ml.bnc.BayesNetClassifierMetadata

class FreqTable private[ml] (
    val counts: Array[Int],
    val featureArity: Vector[Int]) 
extends Serializable {

  def this(featureArity: Vector[Int]) =
    this(
      Array.range(0, featureArity.product).map(_ => 0),
      featureArity
  )

  def getOffsets(featureArity: Vector[Int]): Vector[Int] = featureArity.scanRight(1)(_ * _).tail


  def idx(conf: Vector[Int], offsets: Vector[Int]): Int = conf.zip(offsets).map(t => t._1 * t._2).sum


  val offsets = getOffsets(featureArity)


  def update(conf: Vector[Int], weight: Int): Unit = {
    counts(idx(conf, offsets)) += weight
  } 


  def add(other: FreqTable): FreqTable = {
    require(this.featureArity == other.featureArity, f"Can't add FreqTables, featureArity differs ${this.featureArity} != ${other.featureArity}" )

    val aggCounts: Array[Int] = counts.zip(other.counts).map( t => t._1 + t._2 )
    return new FreqTable(aggCounts, this.featureArity)
  }


  def conditionalProb(X: Vector[Int], Z: Vector[Int]): ProbTable = {

    val M = Vector.range(0, featureArity.length).diff(X ++ Z)

    val counts_smooth = counts.map(_+1)

    val offsetsZ = getOffsets(Z.map(featureArity(_)))

    val baseComb: Vector[Vector[(Int, Int)]] = Vector(Vector())

    val combZ = Z.map(z => Vector.range(0, featureArity(z)).map((z, _)))
    val statesZ = combZ.foldLeft(baseComb)((acc, ids) => for(a <- acc; id <- ids) yield  a :+ id)

    val combX = X.map(x => Vector.range(0, featureArity(x)).map((x, _)))
    val statesX = combX.foldLeft(baseComb)((acc, ids) => for(a <- acc; id <- ids) yield  a :+ id)

    val combM = M.map(m => Vector.range(0, featureArity(m)).map((m, _)))
    val statesM = combM.foldLeft(baseComb)((acc, ids) => for(a <- acc; id <- ids) yield  a :+ id)

    val counts_z = 
      for(zs <- statesZ) 
      yield {
        (for (xs <- statesX; ms <- statesM)
        yield {
            val index = idx(
              (zs ++ xs ++ ms).sortBy(_._1).map(_._2), 
              offsets
            )
            counts_smooth(index)
          }).sum
      }

    if (X.length == 0) {
        val norm = counts_z.sum
        val probs = counts_z.map(_.toDouble/norm)
        return new ProbTable(probs, Z.map(att => featureArity(att)))
    }

    val probs = 
      for (zs <- statesZ; xs <- statesX)
      yield {
          val indexZ = idx(
            zs.map(_._2), // Do not sort this index to preserve the variable order in Z!
            offsetsZ
          )

          (for (ms <- statesM) 
           yield {
              val index = idx(
                (zs ++ xs ++ ms).sortBy(_._1).map(_._2),
                offsets
              )
              counts_smooth(index)      
          }).sum.toDouble / counts_z(indexZ)
      }

    return new ProbTable(probs, (Z ++ X).map(att => featureArity(att)))
  }

  // This is only valid for #(X,Y,C) freq distributions !!
  // TODO: Check feature Cardinality...
  // TODO: Make a general case by marginalizing or cumulating variables
  // Computes MI(X;Y|C)
  //
  def getJointMutualInformation(): Double = {

    var mi = 0.0;

    val card_i = featureArity(0)
    val card_j = featureArity(1)
    val card_c = featureArity(2)

    val counts_c = for (c <- Vector.range(0, card_c))
      yield (for (i <- Vector.range(0, card_i);
                  j <- Vector.range(0, card_j))
        yield counts(i * card_j * card_c + j * card_c + c)).sum

    val counts_ic = (for (i <- Vector.range(0, card_i);
                          c <- Vector.range(0, card_c))
      yield (for (j <- Vector.range(0, card_j))
        yield counts(i * card_j * card_c + j * card_c + c)).reduce(_+_)
      )

    val counts_i =  for (i <- Vector.range(0, card_i))
      yield (for (c <- Vector.range(0, card_c))
        yield counts_ic(i * card_c + c)).reduce(_+_)

    val counts_jc = (for (j <- Vector.range(0, card_j);
                          c <- Vector.range(0, card_c))
      yield (for (i <- Vector.range(0, card_i))
        yield counts(i * card_j * card_c + j * card_c + c)).reduce(_+_)
      )
    val counts_j =  for (j <- Vector.range(0, card_j))
      yield (for (c <- Vector.range(0, card_c))
        yield counts_jc(j * card_c + c)).reduce(_+_)

    val ncases = counts_c.sum

    for (c <- Vector.range(0, card_c)) {
      val p_z = (1.0 * counts_c(c)) / ncases;
      var t_mi = 0.0;

      for(i <- Vector.range(0, card_i);
          j <- Vector.range(0, card_j)) {

        val operator = ( (1.0 * counts(i * card_j * card_c + j * card_c + c)) / (1.0 * counts_c(c)) ) /
          (( (1.0 * counts_ic(i * card_c + c)) / (1.0 * counts_c(c)) ) *
          ( (1.0 * counts_jc(j * card_c + c)) / (1.0 * counts_c(c)) ))

        if (operator > 0)
          t_mi += ( ( (1.0 * counts(i * card_j * card_c + j * card_c + c)) / (1.0 * counts_c(c)) ) ) * Math.log(operator)
      }
      mi += p_z * t_mi;
    }

    mi
  }


  // TODO: check that array is not empty
  def getMutualInformation(x: Int, Z: Vector[Int]): Double = {

    val M = Vector.range(0, featureArity.length).diff(x +: Z)


    val baseComb: Vector[Vector[(Int, Int)]] = Vector(Vector())

    val combZ = Z.map(z => Vector.range(0, featureArity(z)).map((z, _)))
    val statesZ = combZ.foldLeft(baseComb)((acc, ids) => for(a <- acc; id <- ids) yield  a :+ id)

    val combX = Vector(Vector.range(0, featureArity(x)).map((x, _)))
    val statesX = combX.foldLeft(baseComb)((acc, ids) => for(a <- acc; id <- ids) yield  a :+ id)

    val combM = M.map(m => Vector.range(0, featureArity(m)).map((m, _)))
    val statesM = combM.foldLeft(baseComb)((acc, ids) => for(a <- acc; id <- ids) yield  a :+ id)
    

    val counts_x = 
      for(xs <- statesX) 
      yield {
        (for (zs <- statesZ; ms <- statesM)
        yield {
            val index = idx(
              (xs ++ zs ++ ms).sortBy(_._1).map(_._2), 
              offsets
            )
            counts(index)
          }).sum
      }

    val counts_z = 
      for(zs <- statesZ) 
      yield {
        (for (xs <- statesX; ms <- statesM)
        yield {
            val index = idx(
              (zs ++ xs ++ ms).sortBy(_._1).map(_._2), 
              offsets
            )
            counts(index)
          }).sum
      }

    val ncases = counts_z.sum

    val offsets_z = getOffsets(Z.map(featureArity(_)))


    val counts_xz =
      for(zs <- statesZ; xs <- statesX) 
      yield {
        (for (ms <- statesM)
        yield {
            val index = idx(
              (zs ++ xs ++ ms).sortBy(_._1).map(_._2), 
              offsets
            )
            counts(index)
          }).sum
      }

    val offsets_xz = getOffsets((Z :+ x).map(featureArity(_)))


    // MI(x, Z)
    var mii = 0.0;
    for(zs <- statesZ; xs <- statesX) {

      // Do not sort indexes!
      val idx_x = idx( xs.map(_._2), Vector(1) )
      val idx_xz = idx( (zs ++ xs ).map(_._2), offsets_xz )
      val idx_z = idx( (zs).map(_._2), offsets_z )

      val operator = (1.0 * ncases * counts_xz(idx_xz)) / (1.0 * counts_x(idx_x) * counts_z(idx_z))

      if ( operator > 0)
        mii += counts_xz(idx_xz) * Math.log(operator);
    }
    mii = mii / ncases

    mii
  }
 
}


private[spark] object FreqTable {

  // Get the joint frequencies between each attribute and the class
  private[bnc] def computeAttributeFreqs (
    data: RDD[LabeledPoint],
    metadata: BayesNetClassifierMetadata) : RDD[(Vector[Int], FreqTable)] = {

    // We compute iterativelly for each instance and combination as an optimisation
    return data.mapPartitions {
      iterator =>

        // Init attribute | class freq tables 
        val tables: Vector[FreqTable] = for (i <- Vector.range(0, metadata.numFeatures))
          yield new FreqTable(Vector(metadata.featureArity(i), metadata.numClasses))

        // Update pairwise configuration for each instance
        while (iterator.hasNext) {
          iterator.next match { case LabeledPoint(label, features) => {
            // DenseVector must be casted
            val feat = features.toArray.map(_.toInt) 
            // Update the table for each combination
            for (i <- Vector.range(0, metadata.numFeatures)) {
                tables(i).update(Vector(feat(i), label.toInt), 1)
            }
          }}
        }

        // Translate to key value pairs for RDD aggregation
        (for (i <- Vector.range(0, metadata.numFeatures))
          yield (Vector(i), tables(i))
        ).iterator

     // Agregate partial frequencies 
    }.reduceByKey((x: FreqTable, y: FreqTable) => x add y)
  }


  // Get the pairwise joint frequencies between each pair of attributes and the class
  private[bnc] def computeAttributePairwiseFreqs (
    data: RDD[LabeledPoint],
    metadata: BayesNetClassifierMetadata) : RDD[(Vector[Int], FreqTable)] = {

    // We compute iterativelly for each instance and combination as an optimisation
    return data.mapPartitions {
      iterator =>

        // Init pairwise | class freq tables 
        val tables: Vector[FreqTable] = for (i <- Vector.range(0, metadata.numFeatures); j <- Vector.range(i+1, metadata.numFeatures))
          yield new FreqTable(Vector(metadata.featureArity(i), metadata.featureArity(j), metadata.numClasses))

        // Update pairwise configuration for each instance
        while (iterator.hasNext) {
          iterator.next match { case LabeledPoint(label, features) => {
            // DenseVector must be casted
            val feat = features.toArray.map(_.toInt) 
            // Save the order of the combinations by attribute 
            var counter = 0
            // Update the table for each combination
            for (i <- Vector.range(0, metadata.numFeatures); j <- Vector.range(i+1, metadata.numFeatures)) {
                tables(counter).update(Vector(feat(i), feat(j), label.toInt), 1)
                counter += 1
            }
          }}
        }

        // Translate to key value pairs for RDD aggregation
        var counter = -1
        (for (i <- Vector.range(0, metadata.numFeatures); j <- Vector.range(i+1, metadata.numFeatures))
          yield {
            counter += 1
            (Vector(i, j), tables(counter))
          }
        ).iterator

     // Agregate partial frequencies 
    }.reduceByKey((x: FreqTable, y: FreqTable) => x add y)
  }


// Get the triplets joint frequencies between each triplet of attributes and the class
  private[bnc] def computeAttributeX3Freqs (
    data: RDD[LabeledPoint],
    metadata: BayesNetClassifierMetadata) : RDD[(Vector[Int], FreqTable)] = {

    // We compute iterativelly for each instance and combination as an optimisation
    return data.mapPartitions {
      iterator =>

        // Init pairwise | class freq tables 
        val tables: Vector[FreqTable] = 
          for (
            i <- Vector.range(0, metadata.numFeatures); 
            j <- Vector.range(i+1, metadata.numFeatures);
            k <- Vector.range(j+1, metadata.numFeatures))
          yield new FreqTable(Vector(metadata.featureArity(i), metadata.featureArity(j), metadata.featureArity(k), metadata.numClasses))

        // Update pairwise configuration for each instance
        while (iterator.hasNext) {
          iterator.next match { case LabeledPoint(label, features) => {
            // DenseVector must be casted
            val feat = features.toArray.map(_.toInt) 
            // Save the order of the combinations by attribute 
            var counter = 0
            // Update the table for each combination
            for (
              i <- Vector.range(0, metadata.numFeatures); 
              j <- Vector.range(i+1, metadata.numFeatures);
              k <- Vector.range(j+1, metadata.numFeatures)
            ) {
                tables(counter).update(Vector(feat(i), feat(j), feat(k), label.toInt), 1)
                counter += 1
            }
          }}
        }

        // Translate to key value pairs for RDD aggregation
        var counter = -1
        (for (
          i <- Vector.range(0, metadata.numFeatures); 
          j <- Vector.range(i+1, metadata.numFeatures);
          k <- Vector.range(j+1, metadata.numFeatures)
          ) yield {
            counter += 1
            (Vector(i, j, k), tables(counter))
          }
        ).iterator

     // Agregate partial frequencies 
    }.reduceByKey((x: FreqTable, y: FreqTable) => x add y)
  }

private[bnc] def computeAttributeX3FreqsCaped (
    data: RDD[LabeledPoint],
    pairs: Array[(Int, Int)],
    metadata: BayesNetClassifierMetadata) : RDD[(Vector[Int], FreqTable)] = {

    // We compute iterativelly for each instance and combination as an optimisation
    return data.mapPartitions {
      iterator =>

        // Init pairwise | class freq tables 
        val tables: Vector[FreqTable] = pairs.toVector.flatMap { case (i, j) =>
          for (k <- Vector.range(0, metadata.numFeatures) if k != i && k != j)
          yield {
            new FreqTable(Vector(metadata.featureArity(i), metadata.featureArity(j), metadata.featureArity(k), metadata.numClasses))
          }
        } 
          
        // Update pairwise configuration for each instance
        while (iterator.hasNext) {
          iterator.next match { case LabeledPoint(label, features) => {
            // DenseVector must be casted
            val feat = features.toArray.map(_.toInt) 
            // Save the order of the combinations by attribute 
            // Update the table for each combination
            var counter = 0
            pairs.toVector.foreach { case (i, j) =>
              for (k <- Vector.range(0, metadata.numFeatures) if k != i && k != j) {
                  tables(counter).update(Vector(feat(i), feat(j), feat(k), label.toInt), 1)
                  counter += 1
              }
            }
          }}
        }

        // Translate to key value pairs for RDD aggregation
        var counter = -1
        pairs.toVector.flatMap { case (i, j) =>
          for (k <- Vector.range(0, metadata.numFeatures) if k != i && k != j)
          yield {
            counter += 1
            (Vector(i, j, k), tables(counter))
          }
        }.iterator

     // Agregate partial frequencies 
    }.reduceByKey((x: FreqTable, y: FreqTable) => x add y)
  }

// Get the joint frequencies between each attribute a provided parentset and the class
  private[bnc] def computeFreqsParentSet (
    data: RDD[LabeledPoint],
    metadata: BayesNetClassifierMetadata,
    parentSets: Array[Array[Int]]) : RDD[(Vector[Int], FreqTable)] = {

    // We compute iterativelly for each instance and combination as an optimisation
    return data.mapPartitions {
      iterator =>

        // Init attribute | class freq tables 
        val tables: Vector[FreqTable] = for (i <- Vector.range(0, metadata.numFeatures))
          yield {
            val cardinality = metadata.featureArity(i) +: parentSets(i).map(x => metadata.featureArity(x)) :+ metadata.numClasses
            new FreqTable(cardinality.toVector)
          }

        // Update pairwise configuration for each instance
        while (iterator.hasNext) {
          iterator.next match { case LabeledPoint(label, features) => {
            // DenseVector must be casted
            val feat = features.toArray.map(_.toInt) 
            // Update the table for each combination
            for (i <- Vector.range(0, metadata.numFeatures)) {
                val parentValues = parentSets(i).map(x => feat(x))
                val idx = feat(i) +: parentValues :+ label.toInt
                tables(i).update((idx.toVector), 1)
            }
          }}
        }

        // Translate to key value pairs for RDD aggregation
        (for (i <- Vector.range(0, metadata.numFeatures))
          yield (Vector(i), tables(i))
        ).iterator

     // Agregate partial frequencies 
    }.reduceByKey((x: FreqTable, y: FreqTable) => x add y)
  }


  // Get the joint frequencies each combination of attribute a provided parentset and the class
  private[bnc] def computeFreqsGeneric (
    data: RDD[LabeledPoint],
    metadata: BayesNetClassifierMetadata,
    attSets: Array[Vector[Int]]) : RDD[(Vector[Int], FreqTable)] = {

    // We compute iterativelly for each instance and combination as an optimisation
    return data.mapPartitions {
      iterator =>

        // Init attribute | class freq tables 
        val tables: Vector[FreqTable] = attSets.toVector.map { atts =>
          val cardinality = atts.map(x => metadata.featureArity(x)) :+ metadata.numClasses 
          new FreqTable(cardinality.toVector)
        }

        // Update pairwise configuration for each instance
        while (iterator.hasNext) {
          iterator.next match { case LabeledPoint(label, features) => {
            // DenseVector must be casted
            val feat = features.toArray.map(_.toInt) 
            // Update the table for each combination
            attSets.zipWithIndex.foreach { case (atts, i)  =>
                val values = atts.map(x => feat(x))
                val idx = values :+ label.toInt
                tables(i).update((idx.toVector), 1)
            }
          }}
        }

        // Translate to key value pairs for RDD aggregation
        attSets.toVector.zipWithIndex.map { case (atts, i) =>
          (atts.toVector, tables(i))
        }.iterator

     // Agregate partial frequencies 
    }.reduceByKey((x: FreqTable, y: FreqTable) => x add y)
  }
  
}