# Documentation

### Downloading
 
Install Spark 2.4+, Scala 2.11 and Python 3.6+. Spark shell or pyspark shell can be run with maven coordinates
using ``--packages`` argument. This will place all required jars and python files to appropriate executor and driver
paths.

    spark-shell --packages com.github.ozancicek:artan_2.11:0.1.0
    pyspark --packages com.github.ozancicek:artan_2.11:0.1.0
    spark-submit --packages com.github.ozancicek:artan_2.11:0.1.0


For developing with Scala, the dependencies can be retrieved from Maven Central.

    libraryDependencies += "com.github.ozancicek" %% "artan" % "0.1.0"
    
For developing with Python, the dependencies can be installed with pip.

    pip install artan
    
Note that pip will only install the python dependencies, which is not enough to submit jobs to spark cluster. 
To submit pyspark jobs, `--packages='com.github.ozancicek:artan_2.11:0.1.0'` argument should still be specified in
order to download necessary jars from maven central.
 
[Next - Recursive Least Squares](1_recursive_least_squares.md)