import Dependencies._

val defaultScalaVersion = "2.12.8"

scalaVersion := sys.props.getOrElse("scala.version", defaultScalaVersion)

version := "0.1.0-SNAPSHOT"

organization := "com.ozancicek"

organizationName := "ozancicek"

val defaultSparkVersion = "2.4.4"

sparkVersion := sys.props.getOrElse("spark.version", defaultSparkVersion)

lazy val root = (project in file("."))
  .settings(
    name := "artan",
    libraryDependencies += scalaTest % Test,
    fork in run := true)

logBuffered in Test := false

fork in Test := true

javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")

parallelExecution in Test := false

spName := "ozancicek/artan"

spAppendScalaVersion := true

sparkComponents ++= Seq("streaming", "sql", "mllib")
