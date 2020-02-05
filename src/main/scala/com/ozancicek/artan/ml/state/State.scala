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

import java.sql.Timestamp

/**
 * Base trait for state case classes to be used in flatMapGroupsWithState function.
 *
 */
private[state] trait State extends Product {
  val stateIndex: Long
}

/**
 * Base trait for input case classes for flatMapGroupsWithState function. Such case classes should be keyed
 *
 * @tparam KeyType Type of key
 */
private[state] trait KeyedInput[KeyType] extends Product {
  val stateKey: KeyType
  val eventTime: Option[Timestamp]
}

/**
 * Base trait for output case classes after transformation with flatMapGroupsWithState. Such case classes should
 * be keyed
 *
 * @tparam KeyType Type of key
 */
private[state] trait KeyedOutput[KeyType] extends KeyedInput[KeyType] {
  val stateIndex: Long
}