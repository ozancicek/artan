Artan Documentation
###################

    .. toctree::
       :maxdepth: 2
       :caption: Contents:

       index


Setup
*****

Install Spark 2.4+, Scala 2.11 and Python 3.6+. Spark shell or pyspark shell can be run with maven coordinates
using ``--packages`` argument. This will place all required jars and python files to appropriate executor and driver
paths.


        spark-shell --packages com.github.ozancicek:artan_2.11:|artan_version|

        pyspark --packages com.github.ozancicek:artan_2.11:|artan_version|

        spark-submit --packages com.github.ozancicek:artan_2.11:|artan_version|


For developing with Scala, the dependencies can be retrieved from Maven Central.


        libraryDependencies += "com.github.ozancicek" %% "artan" % "|artan_version|"

For developing with Python, the dependencies can be installed with pip.


        pip install artan

Note that pip will only install the python dependencies, which is not enough to submit jobs to spark cluster.
To submit pyspark jobs, ``--packages`` argument to pyspark or spark-submit command should still be specified in
order to download necessary jars from maven central.


        pyspark --packages com.github.ozancicek:artan_2.11:|artan_version|

        spark-submit --packages com.github.ozancicek:artan_2.11:|artan_version|

Guides
******

    .. toctree::
        :maxdepth: 2
        :glob:

        Recursive Least Squares <rlsguide>
        Kalman Filter <lkfguide>
        Extended Kalman Filter <ekfguide>
        Unscented Kalman Filter <ukfguide>
        Multiple-Model Adaptive Filter <mmaeguide>
        stateguide


Python API Reference
********************

    .. toctree::
       :maxdepth: 3
       
       modules

Scaladoc
********

See `scaladoc <https://ozancicek.github.io/docs/scala/artan/latest/index.html#com.github.ozancicek.artan.ml.package>`_

Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
