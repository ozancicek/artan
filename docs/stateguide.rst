Batch - stream compatibility
============================

All functions in this library use the same logic for batch & stream mode, so a state generated with
batch mode is fully compatible with state generated streaming mode. This allows various options regarding
saving & loading a model, for example you can pre-train in batch mode & continue training in stream mode with a
stream-static join.

To start a streaming Kalman filter from a batch state, you need state and stateCovariance values of each state.

    .. code-block:: scala

        // Batch dataframe of measurements
        val batchMeasurements: DataFrame = ...

        val batchFilter = new LinearKalmanFilter(2, 1)
          .setStateKeyCol("stateKey")
          .setMeasurementCol("measurement")

        val batchState = batchFilter.transform(batchMeasurements)
          .filter(s"stateIndex = $batchMeasurementCount")
          .select("stateKey", "state", "stateCovariance").cache()
        batchState.show()
        /*
        +--------+--------------------+--------------------+
        |stateKey|               state|     stateCovariance|
        +--------+--------------------+--------------------+
        |       7|[98.3905315941840...|0.132233902895448...|
        |       3|[99.1890124546266...|0.132233902895448...|
        |       8|[98.7871773752828...|0.132233902895448...|
        |       0|[98.7524328243622...|0.132233902895448...|
        |       5|[98.5564858206287...|0.132233902895448...|
        |       6|[98.8711452158639...|0.132233902895448...|
        |       9|[99.2923263798305...|0.132233902895448...|
        |       1|[98.7803189982662...|0.132233902895448...|
        |       4|[98.9043055447631...|0.132233902895448...|
        |       2|[98.3110820204346...|0.132233902895448...|
        +--------+--------------------+--------------------+
         */

Once this batch state is obtained either from pre-training or a data store, you can do a stream-static
join on stateKey column to get state and stateCovariance columns on the streaming dataframe. Then, you can set
these columns with setInitialStateCol and setInitialCovarianceCol settings and resume training.

    .. code-block:: scala

        // Copy batch filter, except initial state and covariance is read from dataframe column
        val streamFilter = batchFilter
          .setInitialStateCol("state")
          .setInitialCovarianceCol("stateCovariance")

        // Static-stream join to add state & stateCovariance columns.
        val streamMeasurements = streamDF
          .join(batchState, "stateKey")

        val query = streamFilter.transform(streamMeasurements)
          .writeStream
          .queryName("LKFStreamBatchInit")
          .outputMode("append")
          .format("console")
          .start()

        /*
        Batch: 1
        -------------------------------------------
        +--------+----------+--------------------+--------------------+
        |stateKey|stateIndex|               state|     stateCovariance|
        +--------+----------+--------------------+--------------------+
        |       7|         1|[99.7209772179737...|0.132233902867213...|
        |       7|         2|[100.565151317291...|0.132233902623479...|
        |       3|         1|[100.147764225811...|0.132233902867213...|
        |       3|         2|[101.056399834423...|0.132233902623479...|
        |       8|         1|[99.7144109468786...|0.132233902867213...|
        |       8|         2|[100.499087976471...|0.132233902623479...|
        |       0|         1|[99.8782710173084...|0.132233902867213...|
        |       0|         2|[100.700727832003...|0.132233902623479...|
        |       5|         1|[99.4528848590750...|0.132233902867213...|
        |       5|         2|[100.498027806165...|0.132233902623479...|
        |       6|         1|[100.074756380375...|0.132233902867213...|
        |       6|         2|[100.931917973492...|0.132233902623479...|
        |       9|         1|[100.288469838520...|0.132233902867213...|
        |       9|         2|[101.440913991096...|0.132233902623479...|
        |       1|         1|[99.5198257122727...|0.132233902867213...|
        |       1|         2|[100.597885351595...|0.132233902623479...|
        |       4|         1|[99.5943544275477...|0.132233902867213...|
        |       4|         2|[100.529915789434...|0.132233902623479...|
        |       2|         1|[99.4882043828629...|0.132233902867213...|
        |       2|         2|[100.634526656777...|0.132233902623479...|
        +--------+----------+--------------------+--------------------+

         */

See `examples <https://github.com/ozancicek/artan/blob/master/examples/src/main/scala/com/ozancicek/artan/examples/streaming/LKFStreamBatchInit.scala>`_ for the complete code

Restarts
========

In case of a failure or intentional shutdown in streaming mode, spark checkpointing mechanism can be used as usual.


    .. code-block:: scala

        df
          .writeStream
          .outputMode("append")
          .option("checkpointLocation", "path/to/checkpoint/dir")
          .format("memory")
          .start()


The internal state of this library is maintained with avro, so the state will be restored from checkpoints successfully
most of the time. If you make a change that's not allowed by spark (i.e changes listed `here <https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#recovery-semantics-after-changes-in-a-streaming-query>`_)
and need to migrate the state, you can use the pattern in the :ref:`previous section <Batch - stream compatibility>` to recover from
a separate data store.

## Event time and watermarks

## Expiring State

## Version upgrades