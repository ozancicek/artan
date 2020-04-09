Documentation
#############

    .. toctree::
       :maxdepth: 2
       :caption: Contents:

       index


Setup
*****

Install Spark 2.4+, Scala 2.11 and Python 3.6+. Spark shell or pyspark shell can be run with maven coordinates
using ``--packages`` argument. This will place all required jars and python files to appropriate executor and driver
paths.

    .. code-block:: bash

        spark-shell --packages com.github.ozancicek:artan_2.11:0.1.0
        pyspark --packages com.github.ozancicek:artan_2.11:0.1.0
        spark-submit --packages com.github.ozancicek:artan_2.11:0.1.0


For developing with Scala, the dependencies can be retrieved from Maven Central.

    .. code-block:: scala

        libraryDependencies += "com.github.ozancicek" %% "artan" % "0.1.0"

For developing with Python, the dependencies can be installed with pip.

    .. code-block:: bash

        pip install artan

Note that pip will only install the python dependencies, which is not enough to submit jobs to spark cluster.
To submit pyspark jobs, ``--packages`` argument to pyspark or spark-submit command should still be specified in
order to download necessary jars from maven central.

    .. code-block:: bash

        pyspark --packages com.github.ozancicek:artan_2.11:0.1.0
        spark-submit --packages com.github.ozancicek:artan_2.11:0.1.0

Guides
******

    .. toctree::
       :maxdepth: 2
       :glob:

       *guide


Python API Reference
********************

    .. toctree::
       :maxdepth: 3
       
       modules

Scaladoc
********

See `scaladoc <https://ozancicek.github.io/docs/scala/artan/0.2.0-SNAPSHOT/index.html#com.github.ozancicek.artan.ml.package>`_

Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
