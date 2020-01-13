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

package com.ozancicek.artan.ml.state

/**
 * Base trait for state case classes to be used in flatMapGroupsWithState function. Such case classes
 * should be keyed and indexed, and provide a function to convert to an output case class.
 * @tparam KeyType Type of key
 * @tparam OutType Type of output case class
 */
private[state] trait KeyedState[KeyType, OutType <: Product] extends Product {

  val stateKey: KeyType
  val stateIndex: Long

  def asOut: OutType
}

/**
 * Base trait for input case classes for flatMapGroupsWithState function. Such case classes should be keyed
 * @tparam KeyType Type of key
 */
private[state] trait KeyedInput[KeyType] extends Product {

  val stateKey: KeyType
}