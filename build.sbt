import Dependencies._

val defaultScalaVersion = "2.11.8"

val scalaVer = sys.props.getOrElse("scala.version", defaultScalaVersion)

val defaultSparkVersion = "2.4.4"

val sparkVer = sys.props.getOrElse("spark.version", defaultSparkVersion)

val components = Seq("streaming", "sql", "mllib")


lazy val settings = Seq(
  scalaVersion := scalaVer,
  version := "0.1.0",
  organization := "com.github.ozancicek",
  organizationName := "ozancicek",
  sparkVersion := sparkVer,
  libraryDependencies += scalaTest % Test,
  sparkComponents ++= components,
  spAppendScalaVersion := true
)

lazy val root = (project in file("."))
  .settings(
    name := "artan",
    fork in run := true,
    settings)

lazy val examples = (project in file("examples"))
  .settings(
    name := "artan-examples",
    settings)
  .dependsOn(root)

logBuffered in Test := false

fork in Test := true

javaOptions ++= Seq("-Xms512M", "-Xmx2048M", "-XX:MaxPermSize=2048M", "-XX:+CMSClassUnloadingEnabled")

parallelExecution in Test := false

spName := "ozancicek/artan"

credentials += Credentials(Path.userHome / ".ivy2" / ".sparkpackagescredentials")

spShortDescription := "Bayesian filtering with Spark"

spDescription := """Model-parallel bayesian filtering with Apache Spark.
                    |
                    |This library allows you to define model-parallel bayesian filters by leveraging arbitrary stateful
                    |transformation capabilities of Spark DataFrames. Supports both structured streaming and
                    |batch processing mode. Suitable for latent state estimation of many similar small scale systems
                    |rather than a big single system.""".stripMargin

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

