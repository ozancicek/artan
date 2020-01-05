package com.ozancicek.artan.ml.testutils

import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}

trait SparkSessionTestWrapper {
  Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
  lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("test")
      .getOrCreate()
  }

}
