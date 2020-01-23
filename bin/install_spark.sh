#!/usr/bin/env bash

set ex

BUILD="spark-${SPARK_VERSION}-bin-hadoop2.7"

URL="https://dist.apache.org/repos/dist/release/spark/spark-${SPARK_VERSION}/${SPARK_BUILD}.tgz"

wget --quiet $URL -O "/tmp/spark-${SPARK_VERSION}.tgz"
tar -C /opt -xf "/tmp/spark-${SPARK_VERSION}.tgz"
mv /opt/$BUILD "/opt/spark-${SPARK_VERSION}"
rm "/tmp/spark-${SPARK_VERSION}.tgz"

echo "Spark downloaded to"
ls -la "/opt/spark-${SPARK_VERSION}"
