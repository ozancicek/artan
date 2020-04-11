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

Event time and ordering
=======================

If measurements are associated with a timestamp and ordered processing of measurements is desired, event time column
can be set from the input dataframe. This will cause measurements to be processed in ascending order of event time column.

    .. code-block::scala

        // Filter for estimating local linear increasing trend

        val filter = new LinearKalmanFilter(2, 1)
          .setMeasurementCol("measurement")
          .setEventTimeCol("eventTime")
          .setProcessModel(
            new DenseMatrix(2, 2, Array(1, 0, 1, 1)))
          .setProcessNoise(
             new DenseMatrix(2, 2, Array(0.0001, 0.0, 0.0, 0.0001)))
          .setMeasurementNoise(
             new DenseMatrix(1, 1, Array(1)))
          .setMeasurementModel(
            new DenseMatrix(1, 2, Array(1, 0)))

        measurements.show()
        /*
        Shuffled and randomized measurements between 0 ~ 20, eventTime in range 00:00 ~ 03:10
        +--------------------+-------------------+
        |         measurement|          eventTime|
        +--------------------+-------------------+
        |[16.600396246906673]|2010-01-01 02:40:00|
        |[11.642160456918376]|2010-01-01 02:00:00|
        | [5.431510805673608]|2010-01-01 00:50:00|
        |[-0.7805794640849...|2010-01-01 00:00:00|
        | [5.557702620333938]|2010-01-01 01:00:00|
        |[18.480509639899093]|2010-01-01 03:10:00|
        | [8.686614705917332]|2010-01-01 01:30:00|
        |[15.953639806250733]|2010-01-01 02:30:00|
        |[10.292525128550421]|2010-01-01 01:40:00|
        | [18.09287613172998]|2010-01-01 03:00:00|
        |[15.992810861426456]|2010-01-01 02:20:00|
        |[1.1198568487766754]|2010-01-01 00:20:00|
        | [2.336889367434245]|2010-01-01 00:30:00|
        |[0.7527924959565742]|2010-01-01 00:10:00|
        |[17.076684843431103]|2010-01-01 02:50:00|
        |[3.1705195503744017]|2010-01-01 00:40:00|
        |[12.836952969639569]|2010-01-01 02:10:00|
        | [9.909880718374762]|2010-01-01 01:50:00|
        | [7.182921708460937]|2010-01-01 01:10:00|
        | [9.348675648154412]|2010-01-01 01:20:00|
        +--------------------+-------------------+
        */

        filter.transform(measurements).select($"state", $"stateIndex", $"eventTime").show()

        /*
        Measurements processed in sorted order of evenTime, state estimates the trend.
        +--------------------+----------+-------------------+
        |               state|stateIndex|          eventTime|
        +--------------------+----------+-------------------+
        |[-0.7434091904155...|         1|2010-01-01 00:00:00|
        |[0.52340687614430...|         2|2010-01-01 00:10:00|
        |[1.19584880362555...|         3|2010-01-01 00:20:00|
        |[2.22208956892591...|         4|2010-01-01 00:30:00|
        |[3.14516277305829...|         5|2010-01-01 00:40:00|
        |[4.75434051069336...|         6|2010-01-01 00:50:00|
        |[5.71256409043123...|         7|2010-01-01 01:00:00|
        |[6.94015466318830...|         8|2010-01-01 01:10:00|
        |[8.52330679854214...|         9|2010-01-01 01:20:00|
        |[9.35400473381713...|        10|2010-01-01 01:30:00|
        |[10.4189235290492...|        11|2010-01-01 01:40:00|
        |[11.0576627546505...|        12|2010-01-01 01:50:00|
        |[11.9821996520512...|        13|2010-01-01 02:00:00|
        |[12.9725789388260...|        14|2010-01-01 02:10:00|
        |[14.4862990766982...|        15|2010-01-01 02:20:00|
        |[15.6573047856747...|        16|2010-01-01 02:30:00|
        |[16.7165552945660...|        17|2010-01-01 02:40:00|
        |[17.653806140306,...|        18|2010-01-01 02:50:00|
        |[18.6023959572674...|        19|2010-01-01 03:00:00|
        |[19.4403349199569...|        20|2010-01-01 03:10:00|
        +--------------------+----------+-------------------+
        */

Note that **setting event time column will not guarantee end-to-end ordered processing in stream mode**.
Ordering is only guaranteed per minibatch. `Append` output mode is used in the filters,
so if strict ordering in streaming mode is desired aggregated measurements with a specific time window and watermark
should be used as an input to filters. All filters also support setting watermark duration along with event time
column to help propagating watermarks.

    .. code-block:: scala

      val filter = new LinearKalmanFilter(2, 1)
        .setMeasurementCol("measurement")
        .setEventTimeCol("eventTime")
        .setWatermarkDuration("10 seconds")


Expiring State
==============

To cleanup unused state, state timeout can be enabled. Enabling state timeout will clear the state after the
specified timeout duration passes. If a state receives measurements after it times out,
the state will be initialized as if it received no measurements. Supported values are  ``none``,
``process`` and ``event``

*  ``none``: No state timeout, state is kept indefinitely.

* ``process``: Process time based state timeout, state will be cleared if no measurements are received for
    a duration based on processing time. Effects all states. Timeout duration must be set with
    setStateTimeoutDuration.

* ``event``: Event time based state timeout, state will be cleared if no measurements are received for a duration]
    based on event time determined by watermark. Effects all states. Timeout duration must be set with
    setStateTimeoutDuration. Additionally, event time column and it's watermark duration must be set with
    setEventTimeCol and setWatermarkDuration. Note that this will result in dropping measurements occuring later
    than the watermark.

    .. code-block:: scala

        // Event time based state timeout. States receiving no measurements for 12 hours will be cleared.
        // Timeout duration is measured with event time, so event time column must be set.
        val filter = new LinearKalmanFilter(2, 1)
          .setStateKeyCol("modelId")
          .setMeasurementCol("measurement")
          .setEventTimeCol("eventTime")
          .setStateTimeoutDuration("12 hours")
          .setStateTimeoutMode("event")

        // Process time based state timeout. States receiving no measurements for 12 hours will be cleared.
        // Timeout duration is measured with processing time. Therefore, it's not necessary to set event time column
        val filter = new LinearKalmanFilter(2, 1)
          .setStateKeyCol("modelId")
          .setMeasurementCol("measurement")
          .setStateTimeoutDuration("12 hours")
          .setStateTimeoutMode("process")


Version upgrades
================

Semantic versioning is used. In principle, in streaming mode you can update the version of this library without any state
incompatibilities from previously checkpointed state. If a release of this library to cause state incompatibility, this will only
happen in major releases. However, spark version upgrades might render checkpointed state unusable (Due to other stateful
transormations in the code, etc..) so it's always advised to save the state variables in a separate data store
and resume the streaming pipeline using the pattern in the :ref:`previous section <Batch - stream compatibility>`.