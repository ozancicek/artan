import Dependencies._

scalaVersion     := "2.11.8"
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
lazy val testScalastyle = taskKey[Unit]("testScalastyle")
testScalastyle := scalastyle.in(Test).toTask("").value
(test in Test) := ((test in Test) dependsOn testScalastyle).value
