package org.apache.spark.ml

import org.apache.spark.ml.linalg.{VectorUDT, MatrixUDT}

object Types {

  val vectorUDT = new VectorUDT
  val matrixUDT = new MatrixUDT

}
