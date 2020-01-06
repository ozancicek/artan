import Dependencies._

scalaVersion     := "2.12.8"
version          := "0.1.0-SNAPSHOT"
organization     := "com.ozancicek"
organizationName := "ozancicek"
val sparkVersion = "2.4.1"

lazy val root = (project in file("."))
  .settings(
    name := "artan",
    libraryDependencies += scalaTest % Test,
    fork in run := true,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion
    )
  )

logBuffered in Test := false
fork in Test := true
javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")
parallelExecution in Test := false
lazy val compileScalastyle = taskKey[Unit]("compileScalastyle")

compileScalastyle := scalastyle.in(Compile).toTask("").value

(compile in Compile) := ((compile in Compile) dependsOn compileScalastyle).value
(scalastyleConfig in Test) := baseDirectory.value / "scalastyle-config.xml"
