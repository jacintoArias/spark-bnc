package org.apache.spark.ml.bnc

import org.apache.spark.ml.bnc.impl.ProbTable

case class BncNode(att: Int, parents: Vector[Int], table: ProbTable) {

    def nParams(): Int = table.nParams
}