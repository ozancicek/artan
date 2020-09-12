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

package com.github.ozancicek.artan.ml.testutils

import java.io.{File, IOException}

import org.scalatest.Suite
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.scalatest.{BeforeAndAfterAll, Suite}
import java.nio.file.Files

import scala.reflect.io.Directory

/**
 * Base Trait for temp dir, modified from org.apache.spark.ml.util.TempDirectory
 */
trait TempDirectory extends BeforeAndAfterAll { self: Suite =>

  private var _tempDir: File = _
  private var _dir: Directory = _

  /**
   * Returns the temporary directory as a `File` instance.
   */
  protected def tempDir: File = _tempDir

  override def beforeAll(): Unit = {
    super.beforeAll()
    _tempDir = Files.createTempDirectory(this.getClass.getName).toFile
    _dir = new Directory(_tempDir)
  }

  override def afterAll(): Unit = {
    try {
      _dir.deleteRecursively()
    } finally {
      super.afterAll()
    }
  }
}

/**
 * Base Trait for read/write test, modified from org.apache.spark.ml.util.DefaultReadWriteTest
 */
trait DefaultReadWriteTest extends TempDirectory {
  self: Suite =>

  /**
   * Checks "overwrite" option and params.
   * This saves to and loads from [[tempDir]], but creates a subdirectory with a random name
   * in order to avoid conflicts from multiple calls to this method.
   *
   * @param instance   ML instance to test saving/loading
   * @param testParams If true, then test values of Params.  Otherwise, just test overwrite option.
   * @tparam T ML instance type
   * @return Instance loaded from file
   */
  def testDefaultReadWrite[T <: Params with MLWritable](
    instance: T,
    testParams: Boolean = true): T = {
    val uid = instance.uid
    val subdirName = Identifiable.randomUID("test")

    val subdir = new File(tempDir, subdirName)
    val path = new File(subdir, uid).getPath

    instance.save(path)
    intercept[IOException] {
      instance.save(path)
    }
    instance.write.overwrite().save(path)
    val loader = instance.getClass.getMethod("read").invoke(null).asInstanceOf[MLReader[T]]
    val newInstance = loader.load(path)
    assert(newInstance.uid === instance.uid)
    if (testParams) {
      instance.params.foreach { p =>
        if (instance.isDefined(p)) {
          (instance.getOrDefault(p), newInstance.getOrDefault(p)) match {
            case (Array(values), Array(newValues)) =>
              assert(values === newValues, s"Values do not match on param ${p.name}.")
            case (value: Double, newValue: Double) =>
              assert(value.isNaN && newValue.isNaN || value == newValue,
                s"Values do not match on param ${p.name}.")
            case(value: Vector, newValue: Vector) =>
              assert(value === newValue, s"Values do not match on param ${p.name}")
            case(value: Matrix, newValue: Matrix) =>
              assert(value === newValue, s"Values do not match on param ${p.name}")
            case _ =>
          }
        } else {
          assert(!newInstance.isDefined(p), s"Param ${p.name} shouldn't be defined.")
        }
      }
    }

    val load = instance.getClass.getMethod("load", classOf[String])
    val another = load.invoke(instance, path).asInstanceOf[T]
    assert(another.uid === instance.uid)
    another
  }
}