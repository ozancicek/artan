/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ozancicek.artan.ml.testutils

import org.apache.spark.sql._
import org.apache.spark.sql.execution.streaming._

trait StructuredStreamingTestWrapper extends SparkSessionTestWrapper {

  def testAppendQueryAgainstBatch[T: Encoder](
    input: Seq[T],
    queryStream: Dataset[T] => DataFrame,
    queryName: String): Unit = {
    import spark.implicits._
    val streamingResults = runQueryStream(input, "append", queryStream, queryName)
    val batchResults = queryStream(input.toDS()).collect
    streamingResults.zip(batchResults).foreach { case (streamRow, batchRow) => assert(streamRow == batchRow) }
  }

  def runQueryStream[T: Encoder](
    input: Seq[T],
    mode: String,
    queryStream: Dataset[T] => DataFrame,
    queryName: String): Seq[Row] = {

  implicit val sqlContext = spark.sqlContext
  val inputStream = MemoryStream[T]
  val transformed = queryStream(inputStream.toDS())
  val query = transformed.writeStream
    .format("memory")
    .outputMode(mode)
    .queryName(queryName)
    .start()

  inputStream.addData(input)
  query.processAllAvailable()
  val table = spark.table(queryName)
  table.collect.toSeq
  }
}
