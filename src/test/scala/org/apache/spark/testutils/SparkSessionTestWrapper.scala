package org.apache.spark.testutils

import org.apache.spark.sql.SparkSession

trait SparkSessionTestWrapper {

  lazy val spark: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("testSparkSession")
      .getOrCreate()
  }

}
